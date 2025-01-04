import argparse
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
import torch
import torchaudio

def main():
    parser = argparse.ArgumentParser(description="Effectue la diarisation d'un fichier audio avec pyannote.audio.")
    parser.add_argument("input_file", help="Chemin du fichier audio (par ex. default.mp3)")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="Forcer le nombre de locuteurs. Si non spécifié, la pipeline déterminera automatiquement leur nombre.")
    args = parser.parse_args()

    # Récupérer le token Hugging Face depuis la variable d'environnement
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token is None:
        raise ValueError("La variable d'environnement HUGGINGFACE_TOKEN n'est pas définie. Veuillez la définir avant de lancer le script.")

    # Convertir l'audio en WAV temporaire
    audio = AudioSegment.from_file(args.input_file)
    sample_rate = 16000
    audio = audio.set_channels(1).set_frame_rate(sample_rate)  # Mono, 16kHz
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        temp_wav_path = tmp_wav.name
    audio.export(temp_wav_path, format="wav")

    waveform, sample_rate = torchaudio.load(temp_wav_path)
    

    try:
        # Charger la pipeline pré-entraînée de pyannote
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        pipeline.to(torch.device("mps"))

        # Effectuer la diarisation
        if args.num_speakers is not None:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=args.num_speakers)
            #diarization = pipeline(temp_wav_path, num_speakers=args.num_speakers)
        else:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
            #diarization = pipeline(temp_wav_path)

        # Exporter le résultat au format RTTM
        with open("audio.rttm", "w") as rttm:
            diarization.write_rttm(rttm)
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

if __name__ == "__main__":
    main()