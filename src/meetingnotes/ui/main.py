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
                choices=["Local", "MLX", "API"],
                value="Local",
                label="Mode de traitement",
                info="Local: Transformers | MLX: Apple Silicon optimisé | API: Cloud Mistral"
            )
            
            # Choix du modèle selon le mode
            with gr.Row():
                with gr.Column():
                    # Modèles locaux (visible par défaut)
                    local_model_choice = gr.Radio(
                        choices=[
                            "Voxtral-Mini-3B-2507", 
                            "Voxtral-Small-24B-2507"
                        ],
                        value="Voxtral-Mini-3B-2507",
                        label="🤖 Modèle local",
                        info="Mini: Plus rapide | Small: Plus précis, plus de mémoire",
                        visible=True
                    )
                    
                    # Modèles MLX (caché par défaut)
                    mlx_model_choice = gr.Radio(
                        choices=[
                            "Voxtral-Mini-3B-2507",
                            "Voxtral-Small-24B-2507"
                        ],
                        value="Voxtral-Mini-3B-2507",
                        label="🚀 Modèle MLX",
                        info="Mini: Plus rapide | Small: Plus précis, plus de mémoire",
                        visible=False
                    )
                
                with gr.Column():
                    # Précision/quantification (visible pour local et MLX)
                    local_precision_choice = gr.Radio(
                        choices=[
                            "Default",
                            "8bit", 
                            "4bit"
                        ],
                        value="8bit",
                        label="⚡ Précision locale",
                        info="Default: Qualité max | 8bit: Bon compromis | 4bit: Économie mémoire",
                        visible=True
                    )
                    
                    # Précision MLX (caché par défaut)
                    mlx_precision_choice = gr.Radio(
                        choices=[
                            "Default",
                            "8bit",
                            "4bit"
                        ],
                        value="8bit",
                        label="⚡ Précision MLX",
                        info="Default: Qualité max | 8bit: Bon compromis | 4bit: Économie mémoire",
                        visible=False
                    )
                
                # Variables fantômes supprimées
                
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

        # Section diarisation (masquable)
        with gr.Column(elem_classes="processing-section"):
            with gr.Accordion("👥 Identification des locuteurs (optionnel)", open=False):
                gr.Markdown("🔍 **Diarisation automatique** : Analyse des différents locuteurs présents dans l'audio avec pyannote.")
                
                with gr.Row():
                    num_speakers_input = gr.Number(
                        label="👤 Nombre de locuteurs (optionnel)",
                        value=None,
                        minimum=1,
                        maximum=10,
                        info="Laissez vide pour détection automatique",
                        placeholder="Auto"
                    )
                
                btn_diarize = gr.Button(
                    "🎤 Analyser les locuteurs",
                    variant="secondary",
                    size="lg"
                )
                
                
                # Section segments de référence
                gr.Markdown("### 🎵 Segments de référence")
                gr.Markdown("Cliquez sur un locuteur pour écouter son segment de référence :")
                
                speaker_buttons = gr.Radio(
                    label="👥 Locuteurs détectés",
                    choices=[],
                    visible=False,
                    info="Sélectionnez un locuteur pour écouter son segment"
                )
                
                reference_audio_player = gr.Audio(
                    label="🔊 Segment de référence",
                    type="filepath",
                    interactive=False,
                    visible=True
                )
                
                # Section renommage des locuteurs (cachée par défaut)
                with gr.Column(visible=False) as rename_section:
                    gr.Markdown("### ✏️ Renommer un locuteur")
                    
                    with gr.Row():
                        speaker_name_input = gr.Textbox(
                            label="📝 Nouveau nom",
                            placeholder="Entrez le nom du locuteur (ex: Jean, Marie...)",
                            info="Le nom remplacera l'ID du locuteur sélectionné"
                        )
                        
                    btn_apply_rename = gr.Button(
                        "✅ Appliquer tous les renommages",
                        variant="primary",
                        size="sm"
                    )
                    
                    # Indicateur des locuteurs identifiés
                    renamed_speakers_output = gr.Textbox(
                        label="👥 Locuteurs identifiés",
                        value="",
                        lines=5,
                        info="Liste des locuteurs détectés avec leurs noms personnalisés",
                        interactive=False,
                        visible=False
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
            
            # Configuration des sections de résumé
            gr.Markdown("### 📋 Sections du résumé")
            gr.Markdown("Personnalisez les sections à inclure dans votre résumé :")
            
            # Boutons de présélection rapide
            with gr.Row():
                btn_preset_action = gr.Button("🎯 Profil Action", variant="secondary", size="sm")
                btn_preset_info = gr.Button("📊 Profil Information", variant="secondary", size="sm")
                btn_preset_complet = gr.Button("📋 Profil Complet", variant="secondary", size="sm")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**🎯 Sections orientées action**")
                    section_resume_executif = gr.Checkbox(
                        label="📄 Résumé exécutif", 
                        value=True,
                        info="Aperçu global de la réunion"
                    )
                    section_discussions = gr.Checkbox(
                        label="💬 Discussions principales", 
                        value=True,
                        info="Sujets principaux abordés"
                    )
                    section_plan_action = gr.Checkbox(
                        label="✅ Plan d'action", 
                        value=True,
                        info="Actions, responsabilités, échéances"
                    )
                    section_decisions = gr.Checkbox(
                        label="⚖️ Décisions prises", 
                        value=True,
                        info="Décisions validées"
                    )
                    section_prochaines_etapes = gr.Checkbox(
                        label="⏭️ Prochaines étapes", 
                        value=True,
                        info="Actions de suivi"
                    )
                
                with gr.Column():
                    gr.Markdown("**📊 Sections orientées information**")
                    section_sujets_principaux = gr.Checkbox(
                        label="📌 Sujets principaux", 
                        value=False,
                        info="Informations présentées"
                    )
                    section_points_importants = gr.Checkbox(
                        label="⭐ Points importants", 
                        value=False,
                        info="Insights et données clés"
                    )
                    section_questions = gr.Checkbox(
                        label="❓ Questions & discussions", 
                        value=False,
                        info="Questions posées et réponses"
                    )
                    section_elements_suivi = gr.Checkbox(
                        label="📝 Éléments de suivi", 
                        value=False,
                        info="Clarifications nécessaires"
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
            is_local = mode_choice == "Local"
            is_mlx = mode_choice == "MLX"
            is_api = mode_choice == "API"
            return (
                gr.update(visible=is_local),    # local_model_choice
                gr.update(visible=is_local),    # local_precision_choice
                gr.update(visible=is_mlx),      # mlx_model_choice
                gr.update(visible=is_mlx),      # mlx_precision_choice
                gr.update(visible=is_api),      # api_model_choice  
                gr.update(visible=is_api)       # mistral_api_key_direct
            )
        
        processing_mode.change(
            fn=handle_processing_mode_change,
            inputs=[processing_mode],
            outputs=[local_model_choice, local_precision_choice, mlx_model_choice, mlx_precision_choice, api_model_choice, mistral_api_key_direct]
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