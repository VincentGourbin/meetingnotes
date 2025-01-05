# main.py
import os
import gradio as gr

from logic import (
    on_process, 
    on_select_table, 
    on_transcribe, 
    on_summarize
)

def main():
    # Récupérer le token Hugging Face depuis les variables d'environnement
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token is None:
        raise ValueError("La variable d'environnement HUGGINGFACE_TOKEN n'est pas définie.")

    with gr.Blocks() as demo:
        gr.Markdown("# Diarisation, Transcription Complète et Compte Rendu")

        with gr.Row():
            file_input = gr.File(label="Fichier audio/vidéo")
            num_speakers_input = gr.Number(label="Forcer le nombre de locuteurs (optionnel)", value=None)

        # Entrées pour le trimming (en secondes)
        with gr.Row():
            start_trim_input = gr.Number(label="Enlever X secondes au début", value=0)
            end_trim_input = gr.Number(label="Enlever Y secondes à la fin", value=0)

        # Choix de la langue
        language_input = gr.Dropdown(choices=["french", "english"], value="french", label="Langue du meeting")
        
        btn_process = gr.Button("Lancer la diarisation")

        gr.Markdown("### Renommer les Speakers")
        gr.Markdown("Modifiez la colonne `new_speaker` pour changer le nom du speaker. Sélectionnez une ligne pour écouter l'extrait audio associé.")

        with gr.Row():
            old_speakers = gr.Dataframe(
                headers=["old_speaker", "new_speaker"], 
                datatype=["str", "str"], 
                label="Mapping Speaker",
                interactive=True
            )
            audio_excerpt = gr.Audio(label="Extrait du speaker sélectionné", type="filepath")

        # États internes pour stocker segments, wav_path, excerpt_dict
        state_segments = gr.State()
        state_wav_path = gr.State()
        state_excerpt_dict = gr.State()

        # Modèle Whisper
        whisper_model_input = gr.Textbox(label="Modèle Whisper (HF Hub ou chemin local)", value="mlx-community/whisper-large-v3-turbo")
        btn_transcribe = gr.Button("Transcrire")

        # Zone de texte pour la transcription complète
        full_transcription_output = gr.Textbox(label="Transcription complète")

        # Choix du modèle de résumé
        summary_model_input = gr.Dropdown(
            label="Modèle pour le compte rendu",
            choices=[
                "mlx-community/AIDC-AI_Marco-o1_MLX-8bit",
                "mlx-community/Llama-3.3-70B-Instruct-6bit"
            ],
            value="mlx-community/AIDC-AI_Marco-o1_MLX-8bit"
        )

        btn_summarize = gr.Button("Générer le compte rendu")
        summary_output = gr.Textbox(label="Compte Rendu")

        # Événements des boutons et sélections de table
        btn_process.click(
            fn=on_process,
            inputs=[
                file_input, 
                gr.State(value=hf_token), 
                num_speakers_input, 
                start_trim_input, 
                end_trim_input
            ],
            outputs=[
                old_speakers, 
                state_segments, 
                state_wav_path, 
                state_excerpt_dict
            ]
        )
        old_speakers.select(
            fn=on_select_table,
            inputs=[old_speakers, state_excerpt_dict],
            outputs=audio_excerpt
        )

        btn_transcribe.click(
            fn=on_transcribe,
            inputs=[
                old_speakers, 
                state_segments, 
                state_wav_path, 
                whisper_model_input, 
                language_input
            ],
            outputs=[
                full_transcription_output,
                state_segments
            ]
        )

        btn_summarize.click(
            fn=on_summarize,
            inputs=[
                old_speakers, 
                full_transcription_output, 
                state_segments, 
                summary_model_input, 
                language_input
            ],
            outputs=[summary_output]
        )

        # Lancement de l'application
        demo.launch()

if __name__ == "__main__":
    main()