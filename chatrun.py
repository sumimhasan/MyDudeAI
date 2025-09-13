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

@app.route("/chatpage", methods=["GET"])
def chat_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MyDudeAI Chat</title>
<style>
  * { box-sizing: border-box; margin:0; padding:0; font-family: 'Segoe UI', sans-serif;}
  body, html { height:100%; width:100%; }

  body {
    background: #ece5dd;
    display: flex;
  }

.chat-app {
    position: fixed;  /* Fixes it relative to the viewport */
    top: 0;           /* Aligns to the top */
    right: 0;         /* Aligns to the right */
    height: 100vh;
    width: 100%;
    border: 1px solid #000000;
    background: #f3f3f3;
    display: flex;
    flex-direction: column;
    z-index: 1000;    /* Ensures it stays above other content */
}

  /* Header */
  .chat-header {
    height: 60px;
    background: #075e54;
    color: white;
    display: flex;
    align-items: center;
    padding: 0 15px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  }

  /* Messages area */
  .chat-messages {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    background: #d3d1ce;
  }

  .chat-messages::-webkit-scrollbar {
    width: 6px;
  }
  .chat-messages::-webkit-scrollbar-thumb {
    background: rgba(0,0,0,0.2);
    border-radius: 3px;
  }

  /* Message bubbles */
  .message {
    max-width: 70%;
    padding: 10px 15px;
    border-radius: 20px;
    position: relative;
    word-wrap: break-word;
  }

  .user-msg {
    align-self: flex-end;
    background: #dcf8c6;
    border-bottom-right-radius: 0;
  }

  .bot-msg {
    align-self: flex-start;
    background: #fff;
    border-bottom-left-radius: 0;
  }

  /* Input area */
  .chat-input {
    display: flex;
    padding: 10px;
    background: #f0f0f0;
    border-top: 1px solid #ddd;
  }

  .chat-input input {
    flex: 1;
    border-radius: 20px;
    border: none;
    padding: 10px 15px;
    font-size: 16px;
  }

  .chat-input button {
    margin-left: 10px;
    padding: 0 20px;
    background: #075e54;
    color: white;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-weight: bold;
  }

  .chat-input button:hover {
    background: #128c7e;
  }

  /* Typing indicator */
  .typing {
    align-self: flex-start;
    font-size: 14px;
    color: #555;
    font-style: italic;
  }
</style>
</head>
<body>

<div class="chat-app">
  <div class="chat-header">ChatBot</div>
  <div class="chat-messages" id="chatMessages"></div>
  <div class="chat-input">
    <input type="text" id="userInput" placeholder="Type a message..."/>
    <button onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
  const chatMessages = document.getElementById('chatMessages');
  const userInput = document.getElementById('userInput');

  function appendMessage(text, className) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ' + className;
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function botReply(message) {
    let reply = "";

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing';
    typingDiv.textContent = 'Typing...';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;


    fetch('http://localhost:5000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    })
    .then(response => response.json())
    .then(data => {
      chatMessages.removeChild(typingDiv);
      reply = data.reply || "Sorry, I didn't get that.";
      appendMessage(reply, 'bot-msg');
    })
    .catch(() => {
      chatMessages.removeChild(typingDiv);
      appendMessage("Error: Unable to reach the server.", 'bot-msg');
    });
   
  }

  function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    appendMessage(message, 'user-msg');
    userInput.value = '';
    botReply(message);
  }

  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
</script>

</body>
</html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

