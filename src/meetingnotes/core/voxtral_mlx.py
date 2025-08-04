"""
Fonctions principales pour l'analyse directe audio avec Voxtral MLX.

Ce module fournit les fonctions de haut niveau pour :
- L'analyse directe de fichiers audio avec modèles MLX
- La gestion des paramètres et configurations
- L'interface avec l'analyseur MLX Voxtral
"""

import os
import time
from typing import Dict, List, Optional

from ..ai import MemoryManager, cleanup_temp_files
from ..ai.voxtral_mlx_analyzer import VoxtralMLXAnalyzer
from ..utils import format_duration


def on_audio_direct_analysis_mlx(
    wav_path: str,
    model_name: str = "Voxtral-Mini-3B-2507", 
    language: str = "french",
    selected_sections: list = None,
    chunk_duration_minutes: int = 15,
    reference_speakers_data: Optional[str] = None,
    progress_callback=None
) -> Dict[str, str]:
    """
    Analyse directe d'un fichier audio avec Voxtral MLX.
    
    Args:
        wav_path (str): Chemin vers le fichier audio
        model_name (str): Nom du modèle à utiliser
        language (str): Langue du contenu audio
        selected_sections (list): Sections du résumé à inclure
        chunk_duration_minutes (int): Durée des chunks en minutes
        reference_speakers_data (str): Données de diarisation des locuteurs
        
    Returns:
        Dict[str, str]: Résultats avec clé 'transcription'
    """
    print(f"🚀 === Analyse directe audio MLX ===")
    print(f"📂 Fichier: {os.path.basename(wav_path)}")
    print(f"🤖 Modèle: {model_name}")
    print(f"🌍 Langue: {language}")
    print(f"📊 Sections: {selected_sections}")
    print(f"⏱️ Durée chunks: {chunk_duration_minutes}min")
    
    analyzer = None
    try:
        # Créer l'analyseur MLX
        analyzer = VoxtralMLXAnalyzer(model_name=model_name)
        
        # Lancement de l'analyse
        results = analyzer.analyze_audio_chunks_mlx(
            wav_path=wav_path,
            language=language,
            selected_sections=selected_sections,
            chunk_duration_minutes=chunk_duration_minutes,
            reference_speakers_data=reference_speakers_data,
            progress_callback=progress_callback
        )
        
        print(f"✅ Analyse directe MLX terminée avec succès")
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse directe MLX: {e}")
        return {"transcription": f"❌ Erreur analyse MLX: {str(e)}"}
        
    finally:
        # Nettoyer le modèle
        if analyzer:
            try:
                analyzer.cleanup_model()
            except Exception as e:
                print(f"⚠️ Erreur nettoyage modèle MLX: {e}")
        
        # Nettoyage final
        print("🧹 Nettoyage analyse directe MLX terminé")


def on_audio_instruct_summary_mlx(
    file: str,
    model_name: str = "Voxtral-Mini-3B-2507",
    language: str = "french", 
    selected_sections: list = None,
    chunk_duration_minutes: int = 15,
    reference_speakers_data: Optional[str] = None,
    start_trim: float = 0,
    end_trim: float = 0,
    progress_callback=None
) -> Dict[str, str]:
    """
    Interface simplifiée pour l'analyse instruct avec MLX.
    
    Args:
        file (str): Chemin vers le fichier audio
        model_name (str): Nom du modèle
        language (str): Langue
        selected_sections (list): Sections à inclure
        chunk_duration_minutes (int): Durée des chunks
        reference_speakers_data (str): Données diarisation
        start_trim (float): Début à découper (non utilisé pour MLX)
        end_trim (float): Fin à découper (non utilisé pour MLX)
        
    Returns:
        Dict[str, str]: Résultats de l'analyse
    """
    return on_audio_direct_analysis_mlx(
        wav_path=file,
        model_name=model_name,
        language=language,
        selected_sections=selected_sections,
        chunk_duration_minutes=chunk_duration_minutes,
        reference_speakers_data=reference_speakers_data,
        progress_callback=progress_callback
    )