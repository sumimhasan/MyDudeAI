import subprocess
import sys
import threading
import time
from pathlib import Path

# Spinner animation
def spinner(stop_event, message="Processing"):
    symbols = ['.', '..', '...', '.#', '..#', '...#']
    idx = 0
    while not stop_event.is_set():
        print(f"\r{message} {symbols[idx % len(symbols)]} ", end='', flush=True)
        idx += 1
        time.sleep(0.3)
    print("\r", end='', flush=True)

# Install packages quietly
def install_package(package_name):
    stop_event = threading.Event()
    spin_thread = threading.Thread(target=spinner, args=(stop_event, "Installing"))
    spin_thread.start()

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        stop_event.set()
        spin_thread.join()
        print("[SUCCESS] installation for package/dependency")
    except subprocess.CalledProcessError:
        stop_event.set()
        spin_thread.join()
        print("[ERROR]")

# Download HF model
def download_hf_model(model_name, save_dir="models"):
    from transformers import AutoModel, AutoTokenizer
    import torch

    stop_event = threading.Event()
    spin_thread = threading.Thread(target=spinner, args=(stop_event, "Downloading model files"))
    spin_thread.start()

    try:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Load model in FP16
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save locally
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        stop_event.set()
        spin_thread.join()
        print("[SUCCESS] model downloaded and saved ")
    except Exception as e:
        stop_event.set()
        spin_thread.join()
        print(f"[ERROR] failed to download/save model: {e}")

if __name__ == "__main__":
    # Step 1: Install dependencies
    packages = ["transformers", "flask","torch"]  # can add more
    for pkg in packages:
        install_package(pkg)

    print("[INFO] All required libraries and tools installation done!")

    # Step 2: Download HF model
    model_name = "Qwen/Qwen3-1.7B"  # change to your desired model
    download_hf_model(model_name, save_dir="saved_model")

 