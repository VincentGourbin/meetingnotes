import argparse
import os
import json
import tempfile
from pydub import AudioSegment
import mlx_whisper

def main():
    parser = argparse.ArgumentParser(description="Transcrit un fichier audio en tenant compte des segments RTTM et des locuteurs.")
    parser.add_argument("input_file", help="Chemin du fichier audio MP3 (par ex. default.mp3)")
    parser.add_argument("rttm_file", help="Chemin du fichier RTTM (par ex. audio.rttm)")
    parser.add_argument("--output_json", default="transcription.json", help="Nom du fichier JSON de sortie")
    parser.add_argument("--whisper_model", default="mlx-community/whisper-large-v3-turbo",
                        help="Chemin ou repo HF du modèle Whisper (par défaut : mlx-community/whisper-large-v3-turbo)")
    args = parser.parse_args()

    # Chargement et normalisation de l'audio en mémoire
    audio = AudioSegment.from_file(args.input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Mono, 16kHz

    # Lecture du fichier RTTM
    # Format des lignes SPEAKER: SPEAKER <fichier> <canal> <start> <durée> <NA> <NA> <locuteur> <NA> <NA>
    segments_info = []
    with open(args.rttm_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments_info.append((speaker, start, duration))

    # On va créer un tableau final de segments transcrits
    transcribed_segments = []

    # Pour chaque segment, on extrait l'audio correspondant et on le transcrit
    for (speaker, start, duration) in segments_info:
        start_ms = start * 1000
        end_ms = start_ms + (duration * 1000)
        segment_audio = audio[start_ms:end_ms]

        # On doit fournir un fichier à mlx_whisper.transcribe
        # On crée donc un fichier temporaire WAV pour ce segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            seg_path = tmp_wav.name
        segment_audio.export(seg_path, format="wav")

        # Transcription du segment
        result = mlx_whisper.transcribe(
            seg_path,
            path_or_hf_repo=args.whisper_model, 
            language="fr"
        )

        # On récupère la transcription globale du segment
        # result["text"] contient normalement le texte complet transcrit
        text = result.get("text", "").strip()

        # Ajout au tableau final
        transcribed_segments.append({
            "speaker": speaker,
            "start": start,
            "end": start + duration,
            "text": text
        })

        # Nettoyage du fichier temporaire
        if os.path.exists(seg_path):
            os.remove(seg_path)

    # Écriture du JSON de sortie
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"segments": transcribed_segments}, f, ensure_ascii=False, indent=2)

    print(f"Transcription terminée. Résultat écrit dans {args.output_json}")

if __name__ == "__main__":
    main()