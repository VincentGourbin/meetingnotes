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
    extract_audio_from_video,
    handle_diarization,
    handle_speaker_selection,
    handle_speaker_rename
)
from .labels import UILabels

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
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not defined.")

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
        title="MeetingNotes - AI Analysis with Voxtral",
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
        # Main header with style
        with gr.Column(elem_classes="main-header"):
            gr.Markdown(
                f"""
                # {UILabels.MAIN_TITLE}
                {UILabels.MAIN_SUBTITLE}
                {UILabels.MAIN_DESCRIPTION}
                """,
                elem_classes="header-content"
            )

        # Processing mode section (top)
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown(UILabels.PROCESSING_MODE_TITLE)
            
            # Processing mode
            processing_mode = gr.Radio(
                choices=[UILabels.MODE_LOCAL, UILabels.MODE_MLX, UILabels.MODE_API],
                value=UILabels.MODE_LOCAL,
                label=UILabels.PROCESSING_MODE_LABEL,
                info=UILabels.PROCESSING_MODE_INFO
            )
            
            # Model selection according to mode
            with gr.Row():
                with gr.Column():
                    # Local models (visible by default)
                    local_model_choice = gr.Radio(
                        choices=[
                            UILabels.MODEL_MINI, 
                            UILabels.MODEL_SMALL
                        ],
                        value=UILabels.MODEL_MINI,
                        label=UILabels.LOCAL_MODEL_LABEL,
                        info=UILabels.LOCAL_MODEL_INFO,
                        visible=True
                    )
                    
                    # MLX models (hidden by default)
                    mlx_model_choice = gr.Radio(
                        choices=[
                            UILabels.MODEL_MINI,
                            UILabels.MODEL_SMALL
                        ],
                        value=UILabels.MODEL_MINI,
                        label=UILabels.MLX_MODEL_LABEL,
                        info=UILabels.MLX_MODEL_INFO,
                        visible=False
                    )
                
                with gr.Column():
                    # Precision/quantization (visible for local and MLX)
                    local_precision_choice = gr.Radio(
                        choices=[
                            UILabels.PRECISION_DEFAULT,
                            UILabels.PRECISION_8BIT, 
                            UILabels.PRECISION_4BIT
                        ],
                        value=UILabels.PRECISION_8BIT,
                        label=UILabels.LOCAL_PRECISION_LABEL,
                        info=UILabels.LOCAL_PRECISION_INFO,
                        visible=True
                    )
                    
                    # MLX precision (hidden by default)
                    mlx_precision_choice = gr.Radio(
                        choices=[
                            UILabels.PRECISION_DEFAULT,
                            UILabels.PRECISION_8BIT,
                            UILabels.PRECISION_4BIT
                        ],
                        value=UILabels.PRECISION_8BIT,
                        label=UILabels.MLX_PRECISION_LABEL,
                        info=UILabels.MLX_PRECISION_INFO,
                        visible=False
                    )
                
                # API models and key section (hidden by default)
                with gr.Column(visible=False) as api_section:
                    with gr.Row():
                        with gr.Column(scale=1):
                            api_model_choice = gr.Radio(
                                choices=[
                                    UILabels.API_MODEL_MINI,
                                    UILabels.API_MODEL_SMALL
                                ],
                                value=UILabels.API_MODEL_MINI,
                                label=UILabels.API_MODEL_LABEL,
                                info=UILabels.API_MODEL_INFO
                            )
                        with gr.Column(scale=1):
                            mistral_api_key_direct = gr.Textbox(
                                label=UILabels.API_KEY_LABEL,
                                type="password",
                                placeholder=UILabels.API_KEY_PLACEHOLDER,
                                info=UILabels.API_KEY_INFO
                            )

        # Input mode selection
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown(UILabels.INPUT_MODE_TITLE)
            
            input_mode = gr.Radio(
                choices=[UILabels.INPUT_MODE_AUDIO, UILabels.INPUT_MODE_VIDEO], 
                value=UILabels.INPUT_MODE_AUDIO,
                label=UILabels.INPUT_MODE_LABEL,
                info=UILabels.INPUT_MODE_INFO
            )

        # Section Audio (mode par défaut)
        with gr.Column(elem_classes="processing-section") as audio_section:
            gr.Markdown(UILabels.AUDIO_MODE_TITLE)
            
            audio_input = gr.Audio(
                label=UILabels.AUDIO_INPUT_LABEL,
                type="filepath",
                show_label=True,
                interactive=True
            )

        # Section Vidéo (cachée par défaut)
        with gr.Column(elem_classes="processing-section", visible=False) as video_section:
            gr.Markdown(UILabels.VIDEO_MODE_TITLE)
            
            video_input = gr.File(
                label=UILabels.VIDEO_INPUT_LABEL,
                file_types=["video"]
            )
            
            btn_extract_audio = gr.Button(
                UILabels.EXTRACT_AUDIO_BUTTON,
                variant="secondary",
                size="lg"
            )

        # Section options de trim (masquable)
        with gr.Column(elem_classes="processing-section"):
            with gr.Accordion(UILabels.TRIM_OPTIONS_TITLE, open=False):
                with gr.Row():
                    start_trim_input = gr.Number(
                        label=UILabels.START_TRIM_LABEL, 
                        value=0,
                        minimum=0,
                        maximum=3600,
                        info=UILabels.START_TRIM_INFO
                    )
                    end_trim_input = gr.Number(
                        label=UILabels.END_TRIM_LABEL, 
                        value=0,
                        minimum=0,
                        maximum=3600,
                        info=UILabels.END_TRIM_INFO
                    )

        # Section diarisation (masquable)
        with gr.Column(elem_classes="processing-section"):
            with gr.Accordion(UILabels.DIARIZATION_TITLE, open=False):
                gr.Markdown(UILabels.DIARIZATION_DESCRIPTION)
                
                with gr.Row():
                    num_speakers_input = gr.Number(
                        label=UILabels.NUM_SPEAKERS_LABEL,
                        value=None,
                        minimum=1,
                        maximum=10,
                        info=UILabels.NUM_SPEAKERS_INFO,
                        placeholder=UILabels.NUM_SPEAKERS_PLACEHOLDER
                    )
                
                btn_diarize = gr.Button(
                    UILabels.DIARIZE_BUTTON,
                    variant="secondary",
                    size="lg"
                )
                
                
                # Section segments de référence
                gr.Markdown(UILabels.REFERENCE_SEGMENTS_TITLE)
                gr.Markdown(UILabels.REFERENCE_SEGMENTS_DESCRIPTION)
                
                speaker_buttons = gr.Radio(
                    label=UILabels.SPEAKERS_DETECTED_LABEL,
                    choices=[],
                    visible=False,
                    info=UILabels.SPEAKERS_DETECTED_INFO
                )
                
                reference_audio_player = gr.Audio(
                    label=UILabels.REFERENCE_AUDIO_LABEL,
                    type="filepath",
                    interactive=False,
                    visible=True
                )
                
                # Section renommage des locuteurs (cachée par défaut)
                with gr.Column(visible=False) as rename_section:
                    gr.Markdown(UILabels.SPEAKER_RENAME_TITLE)
                    
                    with gr.Row():
                        speaker_name_input = gr.Textbox(
                            label=UILabels.SPEAKER_NAME_LABEL,
                            placeholder=UILabels.SPEAKER_NAME_PLACEHOLDER,
                            info=UILabels.SPEAKER_NAME_INFO
                        )
                        
                    btn_apply_rename = gr.Button(
                        UILabels.APPLY_RENAME_BUTTON,
                        variant="primary",
                        size="sm"
                    )
                    
                    # Indicateur des locuteurs identifiés
                    renamed_speakers_output = gr.Textbox(
                        label=UILabels.IDENTIFIED_SPEAKERS_LABEL,
                        value="",
                        lines=5,
                        info=UILabels.IDENTIFIED_SPEAKERS_INFO,
                        interactive=False,
                        visible=False
                    )

        # Section d'analyse principale
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown(UILabels.MAIN_ANALYSIS_TITLE)
            gr.Markdown(UILabels.MAIN_ANALYSIS_DESCRIPTION)
            
            # Contrôle taille des chunks
            chunk_duration_slider = gr.Slider(
                minimum=5,
                maximum=25,  # Maximum actuel du modèle
                value=15,
                step=5,
                label=UILabels.CHUNK_DURATION_LABEL,
                info=UILabels.CHUNK_DURATION_INFO
            )
            
            # Configuration des sections de résumé
            gr.Markdown(UILabels.SUMMARY_SECTIONS_TITLE)
            gr.Markdown(UILabels.SUMMARY_SECTIONS_DESCRIPTION)
            
            # Boutons de présélection rapide
            with gr.Row():
                btn_preset_action = gr.Button(UILabels.PRESET_ACTION_BUTTON, variant="secondary", size="sm")
                btn_preset_info = gr.Button(UILabels.PRESET_INFO_BUTTON, variant="secondary", size="sm")
                btn_preset_complet = gr.Button(UILabels.PRESET_COMPLETE_BUTTON, variant="secondary", size="sm")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown(UILabels.ACTION_SECTIONS_TITLE)
                    section_resume_executif = gr.Checkbox(
                        label=UILabels.SECTION_EXECUTIVE_SUMMARY, 
                        value=True,
                        info=UILabels.SECTION_EXECUTIVE_SUMMARY_INFO
                    )
                    section_discussions = gr.Checkbox(
                        label=UILabels.SECTION_MAIN_DISCUSSIONS, 
                        value=True,
                        info=UILabels.SECTION_MAIN_DISCUSSIONS_INFO
                    )
                    section_plan_action = gr.Checkbox(
                        label=UILabels.SECTION_ACTION_PLAN, 
                        value=True,
                        info=UILabels.SECTION_ACTION_PLAN_INFO
                    )
                    section_decisions = gr.Checkbox(
                        label=UILabels.SECTION_DECISIONS, 
                        value=True,
                        info=UILabels.SECTION_DECISIONS_INFO
                    )
                    section_prochaines_etapes = gr.Checkbox(
                        label=UILabels.SECTION_NEXT_STEPS, 
                        value=True,
                        info=UILabels.SECTION_NEXT_STEPS_INFO
                    )
                
                with gr.Column():
                    gr.Markdown(UILabels.INFO_SECTIONS_TITLE)
                    section_sujets_principaux = gr.Checkbox(
                        label=UILabels.SECTION_MAIN_TOPICS, 
                        value=False,
                        info=UILabels.SECTION_MAIN_TOPICS_INFO
                    )
                    section_points_importants = gr.Checkbox(
                        label=UILabels.SECTION_KEY_POINTS, 
                        value=False,
                        info=UILabels.SECTION_KEY_POINTS_INFO
                    )
                    section_questions = gr.Checkbox(
                        label=UILabels.SECTION_QUESTIONS, 
                        value=False,
                        info=UILabels.SECTION_QUESTIONS_INFO
                    )
                    section_elements_suivi = gr.Checkbox(
                        label=UILabels.SECTION_FOLLOW_UP, 
                        value=False,
                        info=UILabels.SECTION_FOLLOW_UP_INFO
                    )
            
            btn_direct_transcribe = gr.Button(
                UILabels.ANALYZE_BUTTON, 
                variant="primary",
                size="lg"
            )

        # Section résultats simplifiée
        with gr.Column(elem_classes="results-section"):
            gr.Markdown(UILabels.RESULTS_TITLE)
            
            final_summary_output = gr.Markdown(
                value=UILabels.RESULTS_PLACEHOLDER,
                label=UILabels.RESULTS_LABEL,
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
            is_local = mode_choice == "Local"
            is_mlx = mode_choice == "MLX"
            is_api = mode_choice == "API"
            return (
                gr.update(visible=is_local),    # local_model_choice
                gr.update(visible=is_local),    # local_precision_choice
                gr.update(visible=is_mlx),      # mlx_model_choice
                gr.update(visible=is_mlx),      # mlx_precision_choice
                gr.update(visible=is_api)       # api_section (contient modèle + API key)
            )
        
        processing_mode.change(
            fn=handle_processing_mode_change,
            inputs=[processing_mode],
            outputs=[local_model_choice, local_precision_choice, mlx_model_choice, mlx_precision_choice, api_section]
        )

        # Fonctions de présélection des sections
        def preset_action():
            return (True, True, True, True, True, False, False, False, False)
        
        def preset_info():
            return (True, False, False, False, False, True, True, True, True)
        
        def preset_complet():
            return (True, True, True, True, True, True, True, True, True)
        
        # Gestion de l'analyse directe
        def handle_analysis_direct(
            audio_file, hf_token, language, processing_mode, local_model, local_precision, mlx_model, mlx_precision, api_model, 
            api_key, start_trim, end_trim, chunk_duration,
            s_resume, s_discussions, s_plan_action, s_decisions, s_prochaines_etapes,
            s_sujets_principaux, s_points_importants, s_questions, s_elements_suivi
        ):
            # Construire les paramètres selon le mode
            is_api = processing_mode == "API"
            is_mlx = processing_mode == "MLX"
            
            if is_api:
                # Mode API avec modèle choisi
                transcription_mode = f"API ({api_model})"
                model_key = api_key
            elif is_mlx:
                # Mode MLX avec modèle et précision choisis
                transcription_mode = f"MLX ({mlx_model} ({mlx_precision}))"
                model_key = mlx_model
            else:
                # Mode local avec modèle et précision choisis
                transcription_mode = f"Local ({local_model} ({local_precision}))"
                model_key = local_model
            
            # Récupérer le contexte de diarisation s'il existe
            from .handlers import current_diarization_context
            diarization_data = current_diarization_context if current_diarization_context else None
            
            # Construire la liste des sections sélectionnées
            sections_checkboxes = [
                (s_resume, "resume_executif"),
                (s_discussions, "discussions_principales"), 
                (s_plan_action, "plan_action"),
                (s_decisions, "decisions_prises"),
                (s_prochaines_etapes, "prochaines_etapes"),
                (s_sujets_principaux, "sujets_principaux"),
                (s_points_importants, "points_importants"),
                (s_questions, "questions_discussions"),
                (s_elements_suivi, "elements_suivi")
            ]
            
            selected_sections = [section_key for is_selected, section_key in sections_checkboxes if is_selected]
            
            # Appeler la fonction d'analyse directe avec les sections sélectionnées
            _, summary = handle_direct_transcription(
                audio_file, hf_token, language, transcription_mode,
                model_key, selected_sections, diarization_data, start_trim, end_trim, chunk_duration
            )
            return summary

        # Événements de présélection
        btn_preset_action.click(
            fn=preset_action,
            outputs=[
                section_resume_executif, section_discussions, section_plan_action, 
                section_decisions, section_prochaines_etapes, section_sujets_principaux,
                section_points_importants, section_questions, section_elements_suivi
            ]
        )
        
        btn_preset_info.click(
            fn=preset_info,
            outputs=[
                section_resume_executif, section_discussions, section_plan_action, 
                section_decisions, section_prochaines_etapes, section_sujets_principaux,
                section_points_importants, section_questions, section_elements_suivi
            ]
        )
        
        btn_preset_complet.click(
            fn=preset_complet,
            outputs=[
                section_resume_executif, section_discussions, section_plan_action, 
                section_decisions, section_prochaines_etapes, section_sujets_principaux,
                section_points_importants, section_questions, section_elements_suivi
            ]
        )

        btn_direct_transcribe.click(
            fn=handle_analysis_direct,
            inputs=[
                audio_input,
                gr.State(value=hf_token),
                gr.State("french"),
                processing_mode,
                local_model_choice,
                local_precision_choice,
                mlx_model_choice,
                mlx_precision_choice,
                api_model_choice,
                mistral_api_key_direct,
                start_trim_input,
                end_trim_input,
                chunk_duration_slider,
                section_resume_executif,
                section_discussions,
                section_plan_action,
                section_decisions,
                section_prochaines_etapes,
                section_sujets_principaux,
                section_points_importants,
                section_questions,
                section_elements_suivi
            ],
            outputs=[final_summary_output]
        )

        # Gestion de la diarisation
        btn_diarize.click(
            fn=handle_diarization,
            inputs=[
                audio_input,
                gr.State(value=hf_token),
                num_speakers_input,
                start_trim_input,
                end_trim_input
            ],
            outputs=[speaker_buttons, reference_audio_player, rename_section]
        )
        
        # Gestion de la sélection de locuteur (avec sauvegarde automatique)
        speaker_buttons.change(
            fn=handle_speaker_selection,
            inputs=[speaker_buttons, speaker_name_input],
            outputs=[reference_audio_player, speaker_name_input]
        )
        
        # Gestion du renommage global de tous les locuteurs
        btn_apply_rename.click(
            fn=handle_speaker_rename,
            inputs=[speaker_name_input],
            outputs=[renamed_speakers_output, renamed_speakers_output]
        )

        # Footer avec informations
        with gr.Row():
            gr.Markdown(
                """
                ---
                **MeetingNotes** | Powered by [Voxtral](https://mistral.ai/) | 
                🚀 Intelligent meeting analysis | 💾 Secure local and cloud processing
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