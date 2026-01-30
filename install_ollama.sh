#!/usr/bin/env bash

set -e

echo "========================================"
echo " Installing Ollama + Semantic Parser LLM "
echo "========================================"

# Resolve script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Script location:"
echo "  $SCRIPT_DIR"

echo "[1/5] Updating system..."
sudo apt update

echo "[2/5] Installing system dependencies..."
sudo apt install -y curl python3-pip

echo "[3/5] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "[4/5] Verifying Ollama installation..."
ollama --version

echo "[5/5] Installing Python dependency: httpx..."
pip3 install --user httpx

echo "----------------------------------------"
echo " Creating Ollama model: cd-qwen-big"
echo "----------------------------------------"

MODELFILE_PATH="$SCRIPT_DIR/semantic_parser/Modelfile"

if [ ! -f "$MODELFILE_PATH" ]; then
  echo "ERROR: Modelfile not found."
  echo "Expected location:"
  echo "  $MODELFILE_PATH"
  echo ""
  echo "Please ensure the repository structure is:"
  echo "  <repo_root>/semantic_parser/Modelfile"
  exit 1
fi

ollama create SRL-qwen -f "$MODELFILE_PATH"

echo "----------------------------------------"
echo " Installation complete."
echo " Model 'cd-qwen-big' is ready to use."
echo "----------------------------------------"
