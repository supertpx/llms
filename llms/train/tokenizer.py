import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from tokenizers.implementations import SentencePieceBPETokenizer
import os

random.seed(818)

def train_tokenizer():
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    def get_training_corpus(file_path, batch_size=1000):
        batch = []
        for text in read_texts_from_jsonl(file_path):
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    data_path = './data/mobvoi_seq_monkey_general_open_corpus.jsonl'

    # 初始化tokenizer
    tokenizer = SentencePieceBPETokenizer()

    # 定义特殊token
    special_tokens = ["<unk>", "<pad>", "<mask>", "<s>", "</s>"]

    # 训练tokenizer
    tokenizer.train_from_iterator(
        get_training_corpus(data_path),
        vocab_size=6666,
        special_tokens=special_tokens,  # 确保specialtoken被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<pad>") == 1
    assert tokenizer.token_to_id("<mask>") == 2
    assert tokenizer.token_to_id("<s>") == 3
    assert tokenizer.token_to_id("</s>") == 4

    # 保存tokenizer
    tokenizer_dir = "./model/xmind_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/xmind_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<pad>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<mask>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "3": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "4": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/xmind_tokenizer")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids)
    print('decoder和原始文本是否一致：', response == new_prompt)

def main():
    train_tokenizer()
    # eval_tokenizer()


if __name__ == '__main__':
    main()