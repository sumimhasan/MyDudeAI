#  MyDudeAI

**MyDudeAI** is a **self-hosted AI companion maker** that lets you create, customize, and run your own AI companions locally.  
No cloud lock-in, no external dependencies — you’re in full control.  

---

## ✨ Features

- 🏠 **Self-Hosted** – Run it entirely on your own machine or server.  
- 🎨 **Customizable Companions** – Build unique AI personas with personality, style  
- 🔌 **API First** – Expose REST/WS APIs for integration with your apps aslo have a web ui so you can run on your laptop 
- 🔒 **Privacy Friendly** – Your conversations never leave your system.  
- ⚡ **Fast & Lightweight** – Optimized for local usage with minimal resources just need 4GB or nvidea VRAM GPU  

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/sumimhasan/MyDudeAI.git
cd MyDudeAI
```

### 2. Install dependencies
```bash
python3 setup.py or python setup.py (in windows)
```

### 3. Run locally
```bash
edit the bot configs to make your bot 
python3 botmaker.py
python3 chatrun.py
```

The service will start on [http://localhost:5000](http://localhost:5000).  

---


---

## 📡 API Usage

Example request to send a message to your AI companion:

```bash
web ui -request POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"user": "Hey!", "companionId": "mylocalbot"}'
```
```code
  localhost:5000 --- chat ui
```

---

## 📷 Screenshots / Demo
*(Add images or gifs here once you have UI/preview)*  

---

## 🛠 Tech Stack
- Python, Flask , transformers 
- REST API  
- Local AI model (e.g., QWEN / other)  

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.  

---


---

### 🌟 Support
If you like the project, please **star this repo** ⭐ and share with others!




