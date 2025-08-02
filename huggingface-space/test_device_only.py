#!/usr/bin/env python3
"""
Simple device detection test for MeetingNotes HF Spaces version.
"""

import torch
import os

def test_devices():
    print("🔍 Device Detection Test")
    print("=" * 30)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test device selection logic
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(f"✅ Selected: MPS with {dtype}")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"✅ Selected: CUDA with {dtype}")
    else:
        device = "cpu"
        dtype = torch.float16
        print(f"⚠️ Selected: CPU with {dtype}")
    
    # Test tensor creation on device
    try:
        test_tensor = torch.randn(10, 10, dtype=dtype).to(device)
        print(f"✅ Tensor creation successful on {device}")
        
        if device == "mps":
            print(f"📊 MPS allocated: {torch.mps.current_allocated_memory() / (1024**2):.1f}MB")
        elif device == "cuda":
            print(f"📊 CUDA allocated: {torch.cuda.memory_allocated() / (1024**2):.1f}MB")
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
    
    # Test HF Spaces detection
    is_spaces = os.getenv("SPACE_ID") is not None
    print(f"HF Spaces environment: {is_spaces}")
    
    print("\n🎉 Device detection completed!")

if __name__ == "__main__":
    test_devices()