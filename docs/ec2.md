# EC2 Deployment Guide

This guide covers deploying `sped` on an AWS EC2 GPU instance for production inference with AWQ-quantized models.

## Recommended Instance Types

| Instance | GPU | VRAM | Use Case |
|----------|-----|------|----------|
| `g5.xlarge` | NVIDIA A10G | 24 GB | 7B-13B models (4-bit) |
| `g5.2xlarge` | NVIDIA A10G | 24 GB | 13B-30B models (4-bit) |
| `p3.2xlarge` | NVIDIA V100 | 16 GB | 7B models |
| `g4dn.xlarge` | NVIDIA T4 | 16 GB | 7B models (4-bit) |

## Setup

### 1. Launch EC2 with Ubuntu 22.04

Select an AMI with NVIDIA drivers pre-installed (look for "NVIDIA GPU" or "Deep Learning" AMIs).

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

### 3. Install CUDA (if not pre-installed)

```bash
# Check if CUDA is available
nvcc --version || echo "CUDA not found"

# Install CUDA toolkit (Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

### 4. Install sped

```bash
# Create virtual environment
python3 -m venv ~/sped-env
source ~/sped-env/bin/activate

# Install sped with HF support
pip install sped[hf]

# Verify CUDA is detected
sped info
# Expected: CUDA avail: True
```

### 5. Configure sped

```bash
# Create default config with CUDA target
sped config init
sped config set device cuda
sped config set backend hf
```

Edit `~/.sped/config.yml` with your production settings:

```yaml
backend: hf
device: cuda:0
quantization: 4bit
max_new_tokens: 512
log_level: info
```

## Running as a Daemon

### Option 1: systemd Service

Create `/etc/systemd/system/sped.service`:

```ini
[Unit]
Description=sped Speculative Decoding Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=PATH=/home/ubuntu/sped-env/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/ubuntu/sped-env/bin/sped serve run --target Qwen/Qwen3-0.6B --device cuda
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/sped.log
StandardError=append:/var/log/sped.log

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sped
sudo systemctl start sped
sudo journalctl -u sped -f  # follow logs
```

### Option 2: Using screen/tmux

```bash
tmux new -s sped
sped serve run --target <model> --backend hf --device cuda --log-file ~/sped.log
# Detach: Ctrl+B, D
# Reattach: tmux attach -t sped
```

## Monitoring

### JSON Output for Scripts

```bash
sped serve run --target <model> --prompt "Hello" --output json --device cuda
```

### Log Files

```bash
sped serve run --target <model> --device cuda --log-file /var/log/sped/sped.log
```

### GPU Monitoring

```bash
watch -n 1 nvidia-smi
```

## Benchmarking on EC2

```bash
# Quick benchmark with AWQ model
sped serve run \
  --target Qwen/Qwen3-4B-AWQ \
  --draft Qwen/Qwen3-0.6B \
  --backend hf \
  --device cuda \
  --benchmark \
  --max-new-tokens 32

# Results saved to benchmark_results.json
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce model precision
sped serve run --target <model> --dtype float16 --device cuda

# Use quantization
sped serve run --target <model> --quantize 4bit --device cuda
```

### Model Loading Fails

```bash
# Check cache
sped list models --local

# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--*--*/
```

### Permission Errors

```bash
# Fix home directory permissions
sudo chown -R ubuntu:ubuntu ~/.cache ~/.sped
```

## Security Notes

- Run as a non-root user (e.g., `ubuntu`)
- Use `--log-file` for audit trail
- Restrict API access with firewall rules
- Consider using a reverse proxy (Nginx) for production API serving
