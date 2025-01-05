# wav_converter.py
import os
import tempfile
from pydub import AudioSegment
from moviepy import VideoFileClip

class WavConverter:
    def convert_to_wav(self, input_file: str) -> str:
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