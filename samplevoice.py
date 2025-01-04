import argparse
import os
from pydub import AudioSegment

def main():
    parser = argparse.ArgumentParser(description="Extrait des samples audio de 5s pour chaque locuteur à partir d'un fichier RTTM.")
    parser.add_argument("input_file", help="Chemin du fichier audio (format mp3, par ex. default.mp3)")
    parser.add_argument("rttm_file", help="Chemin du fichier RTTM (par ex. audio.rttm)")
    parser.add_argument("--output_dir", default=".", help="Dossier de sortie pour les échantillons (par défaut le répertoire courant)")
    args = parser.parse_args()

    # Charger l'audio MP3 et le convertir en mono 16kHz en mémoire
    audio = AudioSegment.from_file(args.input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Dictionnaire pour stocker les segments par locuteur
    # locuteurs_segments = { "SPEAKER_00": [(start, duration), ...], ... }
    locuteurs_segments = {}

    # Lecture du fichier RTTM
    with open(args.rttm_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Format: SPEAKER <fichier> <canal> <start> <durée> <NA> <NA> <locuteur> <NA> <NA>
            if parts[0] == "SPEAKER":
                locuteur = parts[7]  # Le locuteur est en colonne 8 (index 7)
                start = float(parts[3])
                duration = float(parts[4])

                if locuteur not in locuteurs_segments:
                    locuteurs_segments[locuteur] = []
                locuteurs_segments[locuteur].append((start, duration))

    # Création du dossier de sortie s'il n'existe pas
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Pour chaque locuteur, on crée un échantillon de 5s
    for locuteur, segments in locuteurs_segments.items():
        # On essaie d'abord de trouver un segment d'au moins 5s
        chosen_segment = None
        for (start, duration) in segments:
            if duration >= 5.0:
                chosen_segment = (start, duration)
                break

        # Si aucun segment n'est assez long, on prend le premier segment, même plus court
        if chosen_segment is None:
            chosen_segment = segments[0]

        seg_start, seg_duration = chosen_segment
        extract_duration = min(seg_duration, 5.0)  # On limite à 5s max

        # Convertir les temps en millisecondes pour pydub
        start_ms = seg_start * 1000
        end_ms = start_ms + (extract_duration * 1000)

        sample_audio = audio[start_ms:end_ms]

        # Nom du fichier de sortie
        output_name = os.path.join(args.output_dir, f"{locuteur}_sample.wav")
        sample_audio.export(output_name, format="wav")
        print(f"Extrait un sample de {extract_duration:.2f} s pour {locuteur} dans {output_name}")

if __name__ == "__main__":
    main()