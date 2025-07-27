"""
Module de conversion audio/vidéo vers le format WAV.

Ce module fournit une classe pour convertir différents formats
audio et vidéo vers un format WAV standardisé (mono, 16kHz).

Dépendances:
    - pydub: Manipulation des fichiers audio
    - moviepy: Extraction audio depuis vidéos
"""

import os
import tempfile
from pydub import AudioSegment
from moviepy import VideoFileClip


class WavConverter:
    """
    Convertisseur de fichiers audio/vidéo vers le format WAV standardisé.
    
    Cette classe gère la conversion de différents formats (MP3, WAV, vidéos)
    vers un format WAV mono 16kHz adapté au traitement vocal.
    """
    def convert_to_wav(self, input_file: str) -> str:
        """
        Convertit un fichier audio/vidéo vers le format WAV standardisé.
        
        Args:
            input_file (str): Chemin vers le fichier d'entrée
        
        Returns:
            str: Chemin vers le fichier WAV temporaire créé
            
        Note:
            Le fichier de sortie est en mono 16kHz, format optimal
            pour les modèles de reconnaissance vocale.
        """
        extension = os.path.splitext(input_file)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            temp_wav_path = tmp_wav.name

        if extension == ".mp3" or extension == ".wav":
            audio = AudioSegment.from_file(input_file)
        else:
            # Traitement vidéo
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_extracted:
                tmp_audio_path = tmp_audio_extracted.name
            clip = VideoFileClip(input_file)
            clip.audio.write_audiofile(tmp_audio_path, codec='pcm_s16le', fps=48000)
            audio = AudioSegment.from_file(tmp_audio_path)
            os.remove(tmp_audio_path)

        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path