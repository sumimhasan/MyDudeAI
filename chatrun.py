from flask import Flask, request, jsonify
import re
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------
# Chat history setup
# --------------------
chat_historyPath = "chat-history/chat-history.json"
try:
    with open(chat_historyPath, 'r') as f:
        chat_history = json.load(f)
except FileNotFoundError:
    chat_history = []

# --------------------
# Model Setup
# --------------------
model_id = "Qwen/Qwen3-1.7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model... (this may take a bit)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=True
)

# --------------------
# Generation Function
# --------------------
INFO_TAG_PATTERN = re.compile(r"<InfoRule>.*?</InfoRule>")

def generate_message(prompt, chat_history, tokenizer, model, 
                     max_new_tokens=1000, temperature=0.8, top_k=20, top_p=0.9,
                     repetition_penalty=1.2, thinking=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chat_history.append({"role": "user", "content": prompt})

    input_ids = tokenizer.apply_chat_template(
        chat_history,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=thinking
    ).to(device)

    generated_ids = input_ids.clone()
    reply_so_far = ""

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                generated_ids,
                attention_mask=(generated_ids != tokenizer.pad_token_id).long().to(device)
            )
            logits = outputs.logits[:, -1, :]

            for token_id in set(generated_ids[0].tolist()):
                logits[0, token_id] /= repetition_penalty

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values,
                                     torch.full_like(logits, float("-inf")),
                                     logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            token_str = tokenizer.decode(next_token[0])

            if not thinking and ("<think>" in token_str or "</think>" in token_str):
                continue

            reply_so_far += token_str

            if next_token.item() == tokenizer.eos_token_id or INFO_TAG_PATTERN.search(reply_so_far):
                break

    chat_history.append({"role": "assistant", "content": reply_so_far})

    # Optional: Save chat history
    with open(chat_historyPath, 'w') as f:
        json.dump(chat_history, f, indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reply_so_far

# --------------------
# Flask API Setup
# --------------------
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]
    response = generate_message(user_message, chat_history, tokenizer, model, device=device)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
