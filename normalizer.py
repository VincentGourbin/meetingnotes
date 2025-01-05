# normalizer.py
import tempfile
from pydub import AudioSegment

class Normalizer:
    def normalize(self, wav_path: str, target_dBFS: float = -20.0) -> str:
        audio = AudioSegment.from_file(wav_path, format="wav")
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_normalized:
            normalized_path = tmp_normalized.name
        normalized_audio.export(normalized_path, format="wav")
        return normalized_path