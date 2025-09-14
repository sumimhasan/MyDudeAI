# 🤖 MyDudeAI

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
git clone https://github.com/YOUR_USERNAME/MyDudeAI.git
cd MyDudeAI
```

### 2. Install dependencies
```bash
npm install
# or
yarn install
```

### 3. Run locally
```bash
npm run dev
```

The service will start on [http://localhost:3000](http://localhost:3000).  

---

## ⚙️ Configuration

You can configure MyDudeAI via `.env` or config files.  
Example:
```env
PORT=3000
DB_PATH=./data/memory.db
MODEL=finestein-4b
```

---

## 📡 API Usage

Example request to send a message to your AI companion:

```bash
web ui -request POST http://localhost:3000/chat \
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
- Node.js / Express (or NestJS if you use it)  
- WebSocket + REST API  
- Local AI model (e.g., FinesteinLM / other)  
- SQLite (default memory DB, pluggable to others)  

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.  

---

## 📜 License
[MIT](./LICENSE)  

---

### 🌟 Support
If you like the project, please **star this repo** ⭐ and share with others!
