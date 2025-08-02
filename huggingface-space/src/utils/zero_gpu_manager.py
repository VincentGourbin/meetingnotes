"""
Zero GPU management for Hugging Face Spaces.

This module provides decorators and utilities for efficient GPU usage
in HF Spaces environment with automatic resource management.
"""

import functools
import gc
import os
import torch
from typing import Callable, Any

# Import spaces if available (HF Spaces environment)
try:
    import spaces
except ImportError:
    spaces = None


class ZeroGPUManager:
    """Manager for Zero GPU operations in HF Spaces."""
    
    def __init__(self):
        # Device selection with MPS support for local Mac testing
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16  # MPS works better with float16
            print("🚀 Using MPS (Apple Silicon) for local testing")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16  # CUDA supports bfloat16
            print("🚀 Using CUDA GPU")
        else:
            self.device = "cpu"
            self.dtype = torch.float16  # CPU with float16 to save memory
            print("⚠️ Using CPU")
        
        self.is_spaces = os.getenv("SPACE_ID") is not None
    
    @staticmethod
    def gpu_task(duration: int = 60):
        """
        Decorator for GPU-intensive tasks.
        
        Args:
            duration: Expected duration in seconds for GPU allocation
        """
        def decorator(func: Callable) -> Callable:
            if spaces is not None and hasattr(spaces, 'GPU'):
                # Use HF Spaces GPU decorator
                return spaces.GPU(duration=duration)(func)
            else:
                # Fallback for local development
                return func
        return decorator
    
    @staticmethod
    def cleanup_gpu():
        """Clean up GPU memory after processing (CUDA/MPS/CPU)."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_device(self) -> str:
        """Get the appropriate device for processing."""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU (CUDA or MPS) is available."""
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    
    def is_spaces_environment(self) -> bool:
        """Check if running in HF Spaces environment."""
        return self.is_spaces
    
    def get_memory_info(self) -> dict:
        """Get current GPU memory information (CUDA or MPS)."""
        if torch.cuda.is_available():
            return {
                "available": True,
                "device": "cuda",
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "total": torch.cuda.get_device_properties(0).total_memory
            }
        elif torch.backends.mps.is_available():
            return {
                "available": True,
                "device": "mps",
                "allocated": torch.mps.current_allocated_memory(),
                "driver_allocated": torch.mps.driver_allocated_memory(),
                # MPS doesn't have total memory info readily available
                "total": "N/A (MPS)"
            }
        else:
            return {"available": False, "device": "cpu"}


# Convenience decorators
def gpu_inference(duration: int = 60):
    """Decorator for GPU inference tasks."""
    return ZeroGPUManager.gpu_task(duration=duration)


def gpu_model_loading(duration: int = 120):
    """Decorator for GPU model loading tasks."""
    return ZeroGPUManager.gpu_task(duration=duration)


def gpu_long_task(duration: int = 300):
    """Decorator for long GPU processing tasks."""
    return ZeroGPUManager.gpu_task(duration=duration)