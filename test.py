from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import torch

import logging
logging.basicConfig(level=logging.INFO)

# 载入预训练模型的分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#special_tokens_dict = {'bos_token': '[BOS]', 'cls_token': '[CLS]', 'eos_token': '[EOS]'}
#tokenizer.add_special_tokens(special_tokens_dict)
# 使用 GPT2Tokenizer 对输入进行编码
text = "<|endoftext|>"
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])
tokens_tensor.shape

def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index

# 读取 GPT-2 预训练模型
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.eval()

total_predicted_text = text
n = 1000  # 预测过程的循环次数
for _ in range(n):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_index = select_top_k(predictions, k=10)
    predicted_text = tokenizer.decode([predicted_index])
    total_predicted_text += tokenizer.decode(predicted_index)

    if predicted_text == '<|endoftext|>':
        # 如果出现文本结束标志，就结束文本生成
        break

    indexed_tokens += [predicted_index]
    tokens_tensor = torch.tensor([indexed_tokens])

print(total_predicted_text)