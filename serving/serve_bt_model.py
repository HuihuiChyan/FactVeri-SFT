import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.
2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.
3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.
2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.
3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]

# Format and tokenize the conversations
conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
conv2_formatted = tokenizer.apply_chat_template(conv2, tokenize=False)
# These two lines remove the potential duplicate bos token
if tokenizer.bos_token is not None and conv1_formatted.startswith(tokenizer.bos_token):
    conv1_formatted = conv1_formatted[len(tokenizer.bos_token):]
if tokenizer.bos_token is not None and conv2_formatted.startswith(tokenizer.bos_token):
    conv2_formatted = conv2_formatted[len(tokenizer.bos_token):]
conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt").to(device)
conv2_tokenized = tokenizer(conv2_formatted, return_tensors="pt").to(device)

# Get the reward scores
with torch.no_grad():
    score1 = rm(**conv1_tokenized).logits[0][0].item()
    score2 = rm(**conv2_tokenized).logits[0][0].item()
print(f"Score for response 1: {score1}")
print(f"Score for response 2: {score2}")

# Expected output:
# Score for response 1: 23.0
# Score for response 2: 3.59375