# Local Installation Guide for Mac

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- Hugging Face account and token

## Installation Steps

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install PyTorch with MPS Support

For the best compatibility, install PyTorch first:

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio
```

### 3. Install Dependencies

```bash
# Install remaining dependencies
pip install -r requirements.txt
```

**Note**: If you encounter issues with `torchaudio`, try:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### 4. Configure Environment

```bash
cp .env.local .env
# Edit .env and add your HF_TOKEN
```

⚠️ **SECURITY WARNING**: 
- `.env` file contains sensitive tokens - NEVER commit it to git!
- `.env` is protected by `.gitignore`
- Only use `.env.local` as a template

### 5. Test Installation

```bash
# Test device detection
python test_device_only.py

# Test full system (if dependencies are working)
python test_local_mac.py

# Start the application
python app.py
```

## Common Issues

### TorchAudio Import Error

If you see Symbol not found errors with torchaudio:

1. **Update PyTorch:**
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

2. **Use CPU version if MPS issues persist:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install from conda (alternative):**
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

### MPS Memory Issues

If you encounter MPS memory errors:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Hugging Face Token

Make sure your HF token has access to:
- `mistralai/Voxtral-Mini-3B-2507`
- `mistralai/Voxtral-Small-24B-2507`
- `pyannote/speaker-diarization-3.1`

## Expected Performance

On Apple Silicon:
- Model loading: 30-60 seconds
- Audio analysis: 2-5x real-time speed
- Memory usage: 4-8GB depending on model size

## Troubleshooting

1. **Check device detection:**
   ```bash
   python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
   ```

2. **Verify dependencies:**
   ```bash
   python -c "import transformers, gradio; print('Dependencies OK')"
   ```

3. **Test MCP server:**
   - Start app with `python app.py`
   - Visit http://localhost:7860
   - Check logs for MCP server initialization

## Support

The application automatically detects and uses the best available device:
- **MPS** (Apple Silicon) - Recommended for Mac
- **CUDA** (NVIDIA GPU) - For PC with GPU
- **CPU** - Fallback option

All features including MCP server work identically across devices.