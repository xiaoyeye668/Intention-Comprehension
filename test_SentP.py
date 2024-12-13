import torch
from transformers import GPT2LMHeadModel, BertTokenizer
model_path = '/Users/xiaoye/Downloads/PROGRAMS/gpt2_surprisal_demo/gpt2-chinese-cluecorpussmall'

def calculate_sentence_probability(context, sentence):
    # 加载GPT-2的tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    # 将上下文和句子拼接起来
    combined_text = context + " " + sentence
    # 对拼接后的文本进行tokenize
    input_ids = tokenizer.encode(combined_text, return_tensors='pt')

    # 用于存储每个token的概率
    probabilities = []
    with torch.no_grad():
        for i in range(len(input_ids[0]) - 1):
            # 获取模型输出
            outputs = model(input_ids[:, :i + 1])
            # 获取最后一个token的预测概率分布（下一个token的概率分布）
            logits = outputs[0][:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            # 获取当前token在概率分布中的概率值
            target_token_id = input_ids[0][i + 1]
            prob = probs[0][target_token_id].item()
            probabilities.append(prob)

    # 计算句子的概率（将每个token的概率相乘）
    sentence_probability = 1
    for p in probabilities:
        sentence_probability *= p

    return sentence_probability

# 示例用法
context = "今天是个好天气"
sentence = "适合出去玩"
probability = calculate_sentence_probability(context, sentence)
print("句子在给定上下文下的概率:", probability)
