#!/usr/bin/env bash

set -e

echo "========================================"
echo " Installing Ollama + Semantic Parser LLM "
echo "========================================"

echo "[1/5] Updating system..."
sudo apt update

echo "[2/5] Installing system dependencies..."
sudo apt install -y curl python3-pip

echo "[3/5] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "[4/5] Verifying Ollama installation..."
ollama --version

echo "[5/5] Installing Python dependency: httpx..."
pip3 install --user httpx --break-system-packages

echo "----------------------------------------"
echo " Creating Ollama model: SRL-qwen"
echo "----------------------------------------"

ollama create SRL-qwen -f Modelfile1

echo "----------------------------------------"
echo " Installation complete."
echo " Model 'SRL-qwen' is ready to use."
echo "----------------------------------------"
