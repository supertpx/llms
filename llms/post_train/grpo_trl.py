import torch

from datasets import load_dataset,Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
import re

import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    PrinterCallback,
)
from Levenshtein import ratio as levenshtein_ratio
import transformers

def create_prompt(sample):
    question = sample['question']
    think_str = " ".join(sample['think'])
    pre_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.  The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think>\n </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>"
    chat = [{"role": "user", "content": pre_prompt + "\n" + question},]
    sample['prompt'] = tokenizer.apply_chat_template(
            conversation=chat,
            tokenize=False,
            add_generation_prompt=True
        )
    sample['think'] = think_str
    return sample

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
    
def format_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think>.*?<answer>(.*?)</answer>.*?$"
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward_func(completions, answer, **kwargs):
    return [1.0 if c == str(gt) else 0.0 for c, gt in zip(completions, answer)]

def levenshtein_reward_func(completions, think, **kwargs):
    res = []
    for completion, sol in zip(completions, think):
        if '</think>' in completion:
            t = completion.split('</think>')[-1]
            res.append(levenshtein_ratio(t, sol))
        else:
            res.append(0.0)
    return res

def length_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """
    根据答案正确度和长短进行奖励的奖励函数
    
    按以下公式计算奖励:
    - 正确答案：reward = 0.5 - (len - min_len)/(max_len - min_len)
    - 错误答案：reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    
    参数:
        completions: 模型生成的完成序列列表
        **kwargs: 其他参数
        
    返回:
        rewards: 每个完成序列的奖励值列表，值在0-0.5之间
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards

def generate_rl(model, text, max_tokens):
    model_input = tokenizer(text, return_tensors='pt').to(model.device)
    model.eval()
    with torch.no_grad():
        tok = model.generate(**model_input, max_new_tokens=max_tokens, pad_token_id=tokenizer.pad_token_type_id)
        outputs = []
        for i in range(len(tok)):
            res = tokenizer.decode(tok[i], skip_special_tokens=True)
            output = res.split(splitter)[-1]
            outputs.append(output)
        return outputs[0] if len(outputs) == 1 else outputs
    
if __name__ == '__main__':

    transformers.set_seed(121)

    # # 加载模型和分词器
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    splitter = '<｜Assistant｜>'
    MAX_STEPS = 250
    USE_PEFT = True
    USE_QLORA = False

    dataset = get_gsm8k_questions()
    # print(dataset)
    dataset = dataset.train_test_split(test_size=0.1)
    print(dataset['train'][2])
    
    if USE_PEFT:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=False,
                )
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        device_map="auto",
                                                        quantization_config=bnb_config,
                                                        trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            device_map="auto",
                                            trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    device_map="auto",
                                                    trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left")

    # dataset = dataset.map(create_prompt)

    reward_functions1 = {'formatting': format_reward_func, 'accuracy': accuracy_reward_func, 'solution_quality': levenshtein_reward_func}
    reward_functions2 = {'soft_format': soft_format_reward_func,
                         'strict_format': strict_format_reward_func, 
                         'xmlcount': xmlcount_reward_func, 
                         'int': int_reward_func,
                         'correctness':correctness_reward_func}

    dtstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    output_directory=f"./DEEPSEEK-GRPO-{dtstr}"


    training_args = GRPOConfig(
        output_dir=output_directory,
        # use_vllm = True,
        # vllm_device = "cuda:1",
        learning_rate=1.e-6,
        lr_scheduler_type = "cosine",
        # optim = "paged_adamw_8bit",
        per_device_train_batch_size=1,
        
        gradient_accumulation_steps=1,
        max_steps=MAX_STEPS,
        
        max_completion_length=1024,
        num_generations=4,
        beta=0.04,
        logging_steps=10,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=MAX_STEPS,
        report_to="none",
        overwrite_output_dir = 'True',    
    )

    if USE_PEFT:
        peft_config = LoraConfig(
            r=32, #Rank
            lora_alpha=32,
            target_modules=[
                "q_proj", 
                "k_proj", 
                "v_proj", 
                "o_proj",
                "gate_proj", 
                "up_proj", 
                "down_proj",
                "dense"
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            # task_type="CAUSAL_LM",
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=list(reward_functions2.values()),
            args=training_args,
            train_dataset=dataset['train'],
            peft_config=peft_config,
            callbacks=[PrinterCallback()]
        )
    else:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=list(reward_functions2.values()),
            args=training_args,
            train_dataset=dataset['train'],
            callbacks=[PrinterCallback()]
        )


    trainer.train()
    
    if USE_PEFT:
        print('Loading trained model')
        CHKPT = MAX_STEPS
        adapter_model_name = f'{output_directory}/checkpoint-{CHKPT}/'
        new_model = PeftModel.from_pretrained(model, adapter_model_name)
    else:
        new_model = model

    user_prompt = '你是谁'
    num_generations = 4
    max_tokens = 1024
    completion = generate_rl(new_model, [user_prompt], max_tokens)
    print("提问：\n", user_prompt)
    print("模型回答：\n", completion)
    
