"""
Module de gestion de la mémoire pour optimiser l'utilisation des ressources.

Ce module fournit des utilitaires pour nettoyer la mémoire GPU/CPU
et éviter l'accumulation excessive lors du traitement de longs fichiers.
"""

import torch
import gc
import os
import psutil
from typing import Optional


class MemoryManager:
    """
    Gestionnaire de mémoire pour les opérations d'IA.
    
    Cette classe fournit des méthodes pour surveiller et nettoyer
    la mémoire utilisée par les modèles de deep learning.
    """
    
    @staticmethod
    def cleanup_gpu_memory():
        """
        Nettoie la mémoire GPU (CUDA ou MPS).
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    
    @staticmethod
    def cleanup_cpu_memory():
        """
        Nettoie la mémoire CPU et force le garbage collection.
        """
        gc.collect()
    
    @staticmethod
    def full_cleanup():
        """
        Nettoyage complet GPU + CPU.
        """
        MemoryManager.cleanup_gpu_memory()
        MemoryManager.cleanup_cpu_memory()
    
    @staticmethod
    def get_memory_stats() -> dict:
        """
        Récupère les statistiques de mémoire actuelles.
        
        Returns:
            dict: Statistiques de mémoire GPU et CPU
        """
        stats = {
            "cpu_percent": psutil.virtual_memory().percent,
            "cpu_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_cached_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        elif torch.backends.mps.is_available():
            stats.update({
                "gpu_allocated_gb": torch.mps.current_allocated_memory() / (1024**3),
                "gpu_type": "MPS (Apple Silicon)"
            })
        
        return stats
    
    @staticmethod
    def print_memory_stats(label: str = ""):
        """
        Affiche les statistiques de mémoire avec un label optionnel.
        
        Args:
            label (str): Label descriptif pour les stats
        """
        stats = MemoryManager.get_memory_stats()
        
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}💾 Mémoire CPU: {stats['cpu_percent']:.1f}% "
              f"(disponible: {stats['cpu_available_gb']:.1f} GB)")
        
        if "gpu_allocated_gb" in stats:
            if "gpu_total_gb" in stats:
                # CUDA
                usage_percent = (stats["gpu_allocated_gb"] / stats["gpu_total_gb"]) * 100
                print(f"{prefix}🎮 GPU CUDA: {stats['gpu_allocated_gb']:.1f}GB / "
                      f"{stats['gpu_total_gb']:.1f}GB ({usage_percent:.1f}%)")
            else:
                # MPS
                print(f"{prefix}🎮 GPU MPS: {stats['gpu_allocated_gb']:.1f}GB alloués")


def cleanup_temp_files(temp_dir: Optional[str] = None):
    """
    Nettoie les fichiers temporaires créés pendant le traitement.
    
    Args:
        temp_dir (str, optional): Répertoire temporaire spécifique à nettoyer
    """
    import tempfile
    import glob
    
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    # Nettoyer les fichiers .wav temporaires
    wav_patterns = [
        os.path.join(temp_dir, "tmp*.wav"),
        os.path.join(temp_dir, "*.wav"),
    ]
    
    cleaned_count = 0
    for pattern in wav_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    cleaned_count += 1
            except (OSError, PermissionError):
                continue  # Ignorer les fichiers en cours d'utilisation
    
    if cleaned_count > 0:
        print(f"🧹 {cleaned_count} fichiers temporaires nettoyés")


# Décorateur pour nettoyage automatique
def auto_cleanup(func):
    """
    Décorateur qui effectue un nettoyage automatique après l'exécution d'une fonction.
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            MemoryManager.full_cleanup()
    return wrapper