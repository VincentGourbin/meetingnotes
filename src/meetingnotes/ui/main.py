"""
Application Gradio pour l'analyse intelligente de réunions avec Voxtral.

Ce module fournit une interface utilisateur web pour analyser des fichiers audio/vidéo
avec l'IA Voxtral de Mistral AI :
1. Analyse directe (transcription + résumé structuré)
2. Modes local et API cloud
3. Support de différents types de réunions

Dépendances:
    - gradio: Interface utilisateur web
    - handlers: Gestionnaires d'événements
    - os: Variables d'environnement

Variables d'environnement requises:
    HUGGINGFACE_TOKEN: Token d'accès pour les modèles Hugging Face
    MISTRAL_API_KEY: Clé API Mistral (optionnelle, pour mode cloud)
"""

import os
import gradio as gr
from dotenv import load_dotenv

from .handlers import (
    handle_direct_transcription,
    handle_input_mode_change,
    extract_audio_from_video
)

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def main():
    """
    Point d'entrée principal de l'application.
    
    Initialise l'interface Gradio pour l'analyse intelligente de réunions
    avec Voxtral (modes local et API cloud).
    
    Raises:
        ValueError: Si la variable d'environnement HUGGINGFACE_TOKEN n'est pas définie
    """
    # Récupérer le token Hugging Face depuis les variables d'environnement
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token is None:
        raise ValueError("La variable d'environnement HUGGINGFACE_TOKEN n'est pas définie.")

    # Configuration du thème Glass personnalisé
    custom_glass_theme = gr.themes.Glass(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        text_size=gr.themes.sizes.text_md,
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_md
    )
    
    with gr.Blocks(
        theme=custom_glass_theme,
        title="MeetingNotes - Analyse IA avec Voxtral",
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .processing-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }
        .results-section {
            margin-top: 25px;
        }
        """
    ) as demo:
        # En-tête principal avec style
        with gr.Column(elem_classes="main-header"):
            gr.Markdown(
                """
                # 🎙️ MeetingNotes
                ### Analyse intelligente de réunions avec IA
                Propulsé par Voxtral
                """,
                elem_classes="header-content"
            )

        # Section mode de traitement (en haut)
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown("### 🎯 Mode de traitement")
            
            # Mode de traitement
            processing_mode = gr.Radio(
                choices=["Local", "API"],
                value="Local",
                label="Mode de traitement",
                info="Local: traitement sur votre machine | API: traitement cloud Mistral"
            )
            
            # Choix du modèle selon le mode
            with gr.Row():
                # Modèles locaux (visible par défaut)
                local_model_choice = gr.Radio(
                    choices=[
                        "Voxtral-Mini-3B-2507", 
                        "Voxtral-Small-24B-2507"
                    ],
                    value="Voxtral-Mini-3B-2507",
                    label="🤖 Modèle local",
                    info="Mini: Plus rapide, moins de mémoire | Small: Plus précis, plus de mémoire",
                    visible=True
                )
                
                # Modèles API (caché par défaut)
                api_model_choice = gr.Radio(
                    choices=[
                        "voxtral-mini-latest",
                        "voxtral-small-latest"
                    ],
                    value="voxtral-mini-latest",
                    label="🌐 Modèle API",
                    info="Mini: Plus rapide, moins cher | Small: Plus précis, plus cher",
                    visible=False
                )

            # Champ API Key (caché par défaut)
            mistral_api_key_direct = gr.Textbox(
                label="🔑 Clé API Mistral",
                type="password",
                placeholder="Entrez votre clé API Mistral...",
                visible=False,
                info="Requise pour utiliser l'API Mistral"
            )

        # Sélection du mode d'entrée
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown("### 📝 Mode d'entrée")
            
            input_mode = gr.Radio(
                choices=["🎵 Audio", "🎬 Vidéo"], 
                value="🎵 Audio",
                label="Type de fichier",
                info="Audio: pour fichiers .wav, .mp3, etc. | Vidéo: pour fichiers .mp4, .avi, etc."
            )

        # Section Audio (mode par défaut)
        with gr.Column(elem_classes="processing-section") as audio_section:
            gr.Markdown("### 🎵 Mode Audio")
            
            audio_input = gr.Audio(
                label="🎙️ Enregistrement ou fichier audio",
                type="filepath",
                show_label=True,
                interactive=True
            )

        # Section Vidéo (cachée par défaut)
        with gr.Column(elem_classes="processing-section", visible=False) as video_section:
            gr.Markdown("### 🎬 Mode Vidéo")
            
            video_input = gr.File(
                label="📁 Fichier vidéo",
                file_types=["video"]
            )
            
            btn_extract_audio = gr.Button(
                "🔄 Extraire l'audio et basculer en mode Audio",
                variant="secondary",
                size="lg"
            )

        # Section options de trim (masquable)
        with gr.Column(elem_classes="processing-section"):
            with gr.Accordion("✂️ Options de découpe (optionnel)", open=False):
                with gr.Row():
                    start_trim_input = gr.Number(
                        label="⏪ Enlever X secondes au début", 
                        value=0,
                        minimum=0,
                        maximum=3600,
                        info="Nombre de secondes à supprimer au début du fichier"
                    )
                    end_trim_input = gr.Number(
                        label="⏩ Enlever X secondes à la fin", 
                        value=0,
                        minimum=0,
                        maximum=3600,
                        info="Nombre de secondes à supprimer à la fin du fichier"
                    )


        # Section d'analyse principale
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown("### ⚡ Analyse de réunion")
            gr.Markdown("💡 **Voxtral IA** : Transcription et résumé structuré intelligent de votre réunion.")
            
            # Contrôle taille des chunks
            chunk_duration_slider = gr.Slider(
                minimum=5,
                maximum=25,  # Maximum actuel du modèle
                value=15,
                step=5,
                label="📦 Taille des morceaux (minutes)",
                info="Durée de chaque chunk audio à traiter séparément"
            )
            
            # Type de réunion
            meeting_type_direct = gr.Radio(
                choices=["Réunion d'information", "Réunion avec plan d'action"],
                value="Réunion d'information",
                label="📋 Type de réunion"
            )
            
            btn_direct_transcribe = gr.Button(
                "⚡ Analyser la réunion", 
                variant="primary",
                size="lg"
            )

        # Section résultats simplifiée
        with gr.Column(elem_classes="results-section"):
            gr.Markdown("### 📋 Résumé de la réunion")
            
            final_summary_output = gr.Markdown(
                value="Le résumé structuré apparaîtra ici après l'analyse...",
                label="📄 Résumé structuré de la réunion",
                height=500
            )


        # Gestion du changement de mode d'entrée
        input_mode.change(
            fn=handle_input_mode_change,
            inputs=[input_mode],
            outputs=[audio_section, video_section]
        )
        
        # Extraction audio depuis vidéo
        btn_extract_audio.click(
            fn=extract_audio_from_video,
            inputs=[video_input, gr.State("french")],
            outputs=[
                audio_input,
                audio_section, 
                video_section,
                input_mode,
                gr.State("french")
            ]
        )

        # Gestion du changement de mode de traitement
        def handle_processing_mode_change(mode_choice):
            is_api = mode_choice == "API"
            return (
                gr.update(visible=not is_api),  # local_model_choice
                gr.update(visible=is_api),      # api_model_choice  
                gr.update(visible=is_api)       # mistral_api_key_direct
            )
        
        processing_mode.change(
            fn=handle_processing_mode_change,
            inputs=[processing_mode],
            outputs=[local_model_choice, api_model_choice, mistral_api_key_direct]
        )

        # Gestion de l'analyse directe
        def handle_analysis_direct(
            audio_file, hf_token, language, processing_mode, local_model, api_model, 
            api_key, meeting_type, start_trim, end_trim, chunk_duration
        ):
            # Construire les paramètres selon le mode
            is_api = processing_mode == "API"
            
            if is_api:
                # Mode API avec modèle choisi
                transcription_mode = f"API ({api_model})"
                model_key = api_key
            else:
                # Mode local avec modèle choisi
                transcription_mode = f"Local ({local_model})"
                model_key = ""
            
            # Appeler la fonction d'analyse directe sans segments de référence
            _, summary = handle_direct_transcription(
                audio_file, hf_token, language, transcription_mode,
                model_key, meeting_type, None, start_trim, end_trim, chunk_duration
            )
            return summary

        btn_direct_transcribe.click(
            fn=handle_analysis_direct,
            inputs=[
                audio_input,
                gr.State(value=hf_token),
                gr.State("french"),
                processing_mode,
                local_model_choice,
                api_model_choice,
                mistral_api_key_direct,
                meeting_type_direct,
                start_trim_input,
                end_trim_input,
                chunk_duration_slider
            ],
            outputs=[final_summary_output]
        )


        # Footer avec informations
        with gr.Row():
            gr.Markdown(
                """
                ---
                **MeetingNotes** | Propulsé par [Voxtral](https://mistral.ai/) | 
                🚀 Analyse intelligente de réunions | 💾 Traitement local et cloud sécurisé
                """,
                elem_classes="footer-info"
            )

        # Lancement de l'application
        demo.launch(
            share=False,
            inbrowser=True,
            show_error=True,
            quiet=False
        )

if __name__ == "__main__":
    main()