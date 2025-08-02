# Troubleshooting Guide

## Common Issues and Solutions

### 1. TorchAudio Import Error on Mac

**Error**: 
```
OSError: dlopen(...libtorchaudio.so, 0x0006): Symbol not found: __ZN3c1015SmallVectorBaseIjE8grow_podEPvmm
```

**Cause**: Version incompatibility between PyTorch and TorchAudio.

**Solutions**:

#### Option A: Reinstall PyTorch Stack (Recommended)
```bash
# Activate your venv
source venv/bin/activate  # or source ../venv/bin/activate

# Uninstall existing PyTorch packages
pip uninstall torch torchvision torchaudio

# Reinstall compatible versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or for nightly builds:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

#### Option B: Use Conda (Alternative)
```bash
# Create new conda environment
conda create -n meetingnotes python=3.10
conda activate meetingnotes

# Install PyTorch with conda
conda install pytorch torchvision torchaudio -c pytorch

# Install other requirements
pip install -r requirements.txt
```

#### Option C: Skip TorchAudio (Testing Only)
If you just want to test the interface without audio processing:
```bash
python simple_test.py  # Basic functionality test
python test_interface_only.py  # UI test without AI models
```

### 2. MCP Server Import Error

**Error**: 
```
ModuleNotFoundError: No module named 'mcp'
```

**Solution**:
```bash
# Install Gradio with MCP support
pip install "gradio[mcp]"
```

### 3. Gradio Version Compatibility

**Error**: 
```
TypeError: Audio.__init__() got an unexpected keyword argument 'info'
```

**Solution**: This is already fixed in the code by removing the `info` parameter and using Markdown for descriptions.

### 4. Gradio Debug Variable Error

**Error**: 
```
ValueError: invalid literal for int() with base 10: 'true'
```

**Solution**: Fix the GRADIO_DEBUG variable in your `.env` file:
```bash
# Change from:
GRADIO_DEBUG=true

# To:
GRADIO_DEBUG=1
```

### 5. Port Already in Use

**Error**: 
```
OSError: Cannot find empty port in range: 7860-7860
```

**Solution**: 
- Kill existing processes: `lsof -ti:7860 | xargs kill -9`
- Or use different port: Set `GRADIO_SERVER_PORT=7861` in your `.env`

### 6. MPS Memory Issues

**Error**: MPS out of memory or crashes

**Solution**:
```bash
# Add to your .env file
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### 7. Hugging Face Token Issues

**Error**: 
```
HTTPError: 401 Client Error: Unauthorized
```

**Solution**:
1. Get a HF token from https://huggingface.co/settings/tokens
2. Add to your `.env`: `HF_TOKEN=your_token_here`
3. Make sure token has access to:
   - `mistralai/Voxtral-Mini-3B-2507`
   - `mistralai/Voxtral-Small-24B-2507`  
   - `pyannote/speaker-diarization-3.1`

## Testing Steps

### 1. Basic Device Test
```bash
python test_device_only.py
```
Expected output:
```
✅ Selected: MPS with torch.float16
✅ Tensor creation successful on mps
```

### 2. Simple Interface Test
```bash
python simple_test.py
```
Should launch a test interface on http://localhost:7862

### 3. Full Interface Test (if dependencies work)
```bash
python test_interface_only.py
```

### 4. Full Application (if all dependencies work)
```bash
python app.py
```

## Environment Setup

### Recommended Setup for Mac
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch first (important!)
pip install torch torchvision torchaudio

# Install Gradio with MCP
pip install "gradio[mcp]"

# Install remaining dependencies
pip install transformers accelerate bitsandbytes
pip install pydub pyannote.audio
pip install python-dotenv httpx numpy scipy

# Configure environment
cp .env.local .env
# Edit .env and add your HF_TOKEN
```

## Getting Help

If issues persist:

1. **Check Python version**: `python --version` (should be 3.9+)
2. **Check PyTorch installation**: `python -c "import torch; print(torch.__version__)"`
3. **Check MPS availability**: `python -c "import torch; print(torch.backends.mps.is_available())"`
4. **Test basic functionality**: `python simple_test.py`

For HF Spaces deployment, these local issues won't affect the production environment which uses CUDA.