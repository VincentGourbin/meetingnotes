"""
Module de normalisation audio.

Ce module fournit une classe pour normaliser le volume
des fichiers audio à un niveau cible en dBFS.

Dépendances:
    - pydub: Manipulation des fichiers audio
"""

import tempfile
from pydub import AudioSegment


class Normalizer:
    """
    Normalisateur de volume audio.
    
    Cette classe permet de normaliser le volume d'un fichier audio
    à un niveau cible pour améliorer la qualité du traitement vocal.
    """
    def normalize(self, wav_path: str, target_dBFS: float = -20.0) -> str:
        """
        Normalise le volume d'un fichier audio.
        
        Args:
            wav_path (str): Chemin vers le fichier WAV d'entrée
            target_dBFS (float): Niveau cible en dBFS (défaut: -20.0)
        
        Returns:
            str: Chemin vers le fichier WAV normalisé
            
        Note:
            Un niveau de -20 dBFS offre un bon équilibre entre
            qualité audio et prévention de la saturation.
        """
        audio = AudioSegment.from_file(wav_path, format="wav")
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_normalized:
            normalized_path = tmp_normalized.name
        normalized_audio.export(normalized_path, format="wav")
        return normalized_path