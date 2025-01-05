# diarizer.py
import os
import torch
import torchaudio
import tempfile
from pydub import AudioSegment
from pyannote.audio import Pipeline

class Diarizer:
    def __init__(self, hf_token: str):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        # Choix du device : MPS (Mac), GPU ou CPU
        device = torch.device("mps") if torch.backends.mps.is_built() else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.pipeline.to(device)

    def diarize(self, wav_path: str, num_speakers: int = None):
        waveform, sample_rate = torchaudio.load(wav_path)
        if num_speakers is not None:
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                num_speakers=num_speakers
            )
        else:
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}
            )
        return diarization

def extract_speaker_snippet(segments, wav_path, speaker):
    """
    Extrait le PREMIER snippet d'au moins 3 secondes pour un speaker donné,
    puis coupe le segment à 3 secondes si jamais il est plus long.
    """
    # Filtrer les segments du speaker
    speaker_segments = [seg for seg in segments if seg["speaker"] == speaker]
    if not speaker_segments:
        return None

    audio = AudioSegment.from_file(wav_path, format="wav")
    three_seconds = 3000  # 3 s en millisecondes
    chosen_seg = None

    # On cherche le premier segment >= 3 secondes
    for seg in speaker_segments:
        length_ms = (seg["end"] - seg["start"]) * 1000
        if length_ms >= three_seconds:
            chosen_seg = seg
            break

    # Si aucun segment ne fait au moins 3 secondes, on renvoie None
    if not chosen_seg:
        return None

    start_ms = chosen_seg["start"] * 1000
    end_ms = chosen_seg["end"] * 1000
    excerpt_length = end_ms - start_ms

    # On se limite à 3 secondes si le segment est plus long
    if excerpt_length > three_seconds:
        end_ms = start_ms + three_seconds

    excerpt = audio[start_ms:end_ms]

    # Sauvegarde temporaire du snippet
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_excerpt:
        excerpt_path = tmp_excerpt.name
    excerpt.export(excerpt_path, format="wav")

    return excerpt_path