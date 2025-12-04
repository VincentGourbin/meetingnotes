"""
Application Gradio pour l'analyse intelligente de réunions avec Voxtral - Version HF Spaces.

Version adaptée pour Hugging Face Spaces avec :
- Uniquement mode Transformers (MLX et API supprimés)
- Modèles 8-bit uniquement
- Support MCP natif
- Zero GPU decorators
"""

import os
import gradio as gr
from dotenv import load_dotenv

from ..ai.voxtral_spaces_analyzer import VoxtralSpacesAnalyzer
from ..utils.zero_gpu_manager import ZeroGPUManager, gpu_inference

# Import labels locally
from .labels import UILabels

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Global instances for MCP functions
analyzer = None
gpu_manager = None

def initialize_components():
    """Initialize global components for MCP functions."""
    global analyzer, gpu_manager
    if analyzer is None:
        analyzer = VoxtralSpacesAnalyzer()
        gpu_manager = ZeroGPUManager()

# MCP Tools - exposed automatically by Gradio
@gpu_inference(duration=300)
def analyze_meeting_audio(
    audio_file: str,
    sections: list = None,
    model_name: str = "Voxtral-Mini-3B-2507"
) -> dict:
    """
    Analyze meeting audio and generate structured summaries using Voxtral AI.
    
    This function processes audio files to extract insights and generate 
    structured meeting summaries with configurable sections.
    
    Args:
        audio_file: Path to the audio file to analyze (MP3, WAV, M4A, OGG)
        sections: List of analysis sections to include (executive_summary, action_plan, etc.)
        model_name: Voxtral model to use for analysis (Mini-3B or Small-24B)
    
    Returns:
        Dictionary containing analysis results, processing time, and metadata
    """
    initialize_components()
    
    if not os.path.exists(audio_file):
        return {"error": "Audio file not found", "status": "failed"}
    
    try:
        import time
        start_time = time.time()
        
        # Set default sections if none provided
        if sections is None:
            sections = ["resume_executif", "discussions_principales", "plan_action"]
        
        # Switch model if different
        if analyzer.current_model_key != model_name:
            analyzer.switch_model(model_name)
        
        # Analyze audio (MCP function without progress bar)
        results = analyzer.analyze_audio_chunks(
            wav_path=audio_file,
            language="auto",
            selected_sections=sections
        )
        
        processing_time = time.time() - start_time
        
        return {
            "status": "completed",
            "analysis": results.get("transcription", "No analysis available"),
            "processing_time_seconds": processing_time,
            "model_used": model_name,
            "sections_analyzed": sections
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "processing_time_seconds": time.time() - start_time if 'start_time' in locals() else 0
        }
    finally:
        if gpu_manager:
            gpu_manager.cleanup_gpu()

def get_available_sections() -> dict:
    """Get available analysis sections for meeting summaries."""
    from ..ai.prompts_config import VoxtralPrompts
    return {
        "status": "success",
        "sections": VoxtralPrompts.AVAILABLE_SECTIONS,
        "total_sections": len(VoxtralPrompts.AVAILABLE_SECTIONS)
    }

def get_meeting_templates() -> dict:
    """Get pre-configured meeting analysis templates."""
    templates = {
        "action_meeting": {
            "name": "Action-Oriented Meeting",
            "description": "For meetings focused on decisions and action items",
            "recommended_sections": ["resume_executif", "discussions_principales", "plan_action", "decisions_prises", "prochaines_etapes"]
        },
        "info_meeting": {
            "name": "Information Meeting", 
            "description": "For presentations and informational sessions",
            "recommended_sections": ["resume_executif", "sujets_principaux", "points_importants", "questions_discussions", "elements_suivi"]
        }
    }
    return {"status": "success", "templates": templates, "total_templates": len(templates)}

# Handlers adaptés pour HF Spaces
def handle_input_mode_change(input_mode):
    """Gestion du changement de mode d'entrée."""
    if input_mode == UILabels.INPUT_MODE_AUDIO:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def extract_audio_from_video(video_file, language):
    """Extraction audio depuis vidéo (placeholder pour HF Spaces)."""
    if video_file is None:
        return None, gr.update(visible=True), gr.update(visible=False), UILabels.INPUT_MODE_AUDIO, language
    
    # Pour HF Spaces, on assume que le processing vidéo sera fait côté client
    # ou qu'on accepte déjà des fichiers audio
    return video_file, gr.update(visible=True), gr.update(visible=False), UILabels.INPUT_MODE_AUDIO, language


@gpu_inference(duration=300)
def handle_direct_transcription(
    audio_file, hf_token, language, transcription_mode, model_key, 
    selected_sections, start_trim, end_trim, progress=gr.Progress()
):
    """Gestion de l'analyse directe adaptée pour HF Spaces."""
    initialize_components()
    
    if audio_file is None:
        return "", "❌ Veuillez d'abord télécharger un fichier audio."
    
    try:
        # Extraire le nom du modèle depuis transcription_mode
        if "Mini" in transcription_mode:
            model_name = "Voxtral-Mini-3B-2507"
        else:
            model_name = "Voxtral-Small-24B-2507"
        
        # Configurer l'analyseur
        if analyzer.current_model_key != model_name:
            analyzer.switch_model(model_name)
        
        # Setup progress callback
        def progress_callback(progress_ratio, message):
            progress(progress_ratio, desc=message)
        
        # Lancer l'analyse (chunk duration automatique selon le modèle)
        results = analyzer.analyze_audio_chunks(
            wav_path=audio_file,
            language="auto",
            selected_sections=selected_sections,
            start_trim=start_trim,
            end_trim=end_trim,
            progress_callback=progress_callback
        )
        
        return "", results.get("transcription", "Aucune analyse disponible")
        
    except Exception as e:
        error_msg = f"❌ Erreur lors de l'analyse: {str(e)}"
        return "", error_msg
    finally:
        if gpu_manager:
            gpu_manager.cleanup_gpu()

def create_spaces_interface():
    """
    Point d'entrée principal pour l'interface HF Spaces.
    
    Interface identique au projet original mais simplifiée :
    - Seul mode Transformers (pas MLX/API)
    - Modèles pré-quantisés uniquement 
    - Support MCP natif
    """
    # Initialize components
    initialize_components()
    
    # Récupérer le token Hugging Face depuis les variables d'environnement
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token is None:
        print("⚠️ Warning: HF_TOKEN environment variable not found")

    # Configuration du thème Glass personnalisé (identique à l'original)
    custom_glass_theme = gr.themes.Glass(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        text_size=gr.themes.sizes.text_md,
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_md
    )

    # CSS personnalisé pour l'application
    custom_css = """
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

    with gr.Blocks(
        title="MeetingNotes - AI Analysis with Voxtral"
    ) as demo:
        # Main header with style (identique à l'original)
        with gr.Column(elem_classes="main-header"):
            gr.Markdown(
                f"""
                # {UILabels.MAIN_TITLE}
                {UILabels.MAIN_SUBTITLE}
                {UILabels.MAIN_DESCRIPTION}
                """,
                elem_classes="header-content"
            )

        # Processing mode section (SIMPLIFIÉ - seulement Transformers)
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown("## 🔧 Model Configuration")
            
            # Model selection (modèles pré-quantisés)
            local_model_choice = gr.Radio(
                choices=[UILabels.MODEL_MINI, UILabels.MODEL_SMALL],
                value=UILabels.MODEL_MINI,
                label="Voxtral Model Selection"
            )
            
            # Information about the models
            gr.Markdown("""
            **📋 About this HF Spaces version:**
            - Uses standard Mistral Voxtral models optimized for Zero GPU
            - **Mini Model**: [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) - Faster processing, lower memory usage
            - **Small Model**: [Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) - Higher quality analysis, more detailed summaries
            - Chunk duration automatically optimized: 15min for Mini, 10min for Small
            
            **🔗 Complete version available:**
            For local processing (MLX/Transformers), API modes, and **speaker diarization**, check the full version on [GitHub](https://github.com/VincentGourbin/meetingnotes)
            """)

        # Input mode selection (identique à l'original)
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown(UILabels.INPUT_MODE_TITLE)
            
            input_mode = gr.Radio(
                choices=[UILabels.INPUT_MODE_AUDIO, UILabels.INPUT_MODE_VIDEO], 
                value=UILabels.INPUT_MODE_AUDIO,
                label=UILabels.INPUT_MODE_LABEL
            )

        # Section Audio (mode par défaut) - identique à l'original
        with gr.Column(elem_classes="processing-section") as audio_section:
            gr.Markdown(UILabels.AUDIO_MODE_TITLE)
            
            audio_input = gr.Audio(
                label=UILabels.AUDIO_INPUT_LABEL,
                type="filepath",
                show_label=True,
                interactive=True
            )

        # Section Vidéo (cachée par défaut) - identique à l'original
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

        # Section options de trim (identique à l'original)
        with gr.Column(elem_classes="processing-section"):
            with gr.Accordion(UILabels.TRIM_OPTIONS_TITLE, open=False):
                with gr.Row():
                    start_trim_input = gr.Number(
                        label=UILabels.START_TRIM_LABEL, 
                        value=0,
                        minimum=0,
                        maximum=3600
                    )
                    end_trim_input = gr.Number(
                        label=UILabels.END_TRIM_LABEL, 
                        value=0,
                        minimum=0,
                        maximum=3600
                    )


        # Section d'analyse principale (identique à l'original)
        with gr.Column(elem_classes="processing-section"):
            gr.Markdown(UILabels.MAIN_ANALYSIS_TITLE)
            gr.Markdown(UILabels.MAIN_ANALYSIS_DESCRIPTION)
            gr.Markdown("*Chunk duration is automatically optimized: 15min for Mini, 10min for Small (Zero GPU optimization)*")
            
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
                    section_resume_executif = gr.Checkbox(label=UILabels.SECTION_EXECUTIVE_SUMMARY, value=True)
                    section_discussions = gr.Checkbox(label=UILabels.SECTION_MAIN_DISCUSSIONS, value=True)
                    section_plan_action = gr.Checkbox(label=UILabels.SECTION_ACTION_PLAN, value=True)
                    section_decisions = gr.Checkbox(label=UILabels.SECTION_DECISIONS, value=True)
                    section_prochaines_etapes = gr.Checkbox(label=UILabels.SECTION_NEXT_STEPS, value=True)
                
                with gr.Column():
                    gr.Markdown(UILabels.INFO_SECTIONS_TITLE)
                    section_sujets_principaux = gr.Checkbox(label=UILabels.SECTION_MAIN_TOPICS, value=False)
                    section_points_importants = gr.Checkbox(label=UILabels.SECTION_KEY_POINTS, value=False)
                    section_questions = gr.Checkbox(label=UILabels.SECTION_QUESTIONS, value=False)
                    section_elements_suivi = gr.Checkbox(label=UILabels.SECTION_FOLLOW_UP, value=False)
            
            btn_direct_transcribe = gr.Button(
                UILabels.ANALYZE_BUTTON, 
                variant="primary",
                size="lg"
            )

        # Section résultats (identique à l'original)
        with gr.Column(elem_classes="results-section"):
            gr.Markdown(UILabels.RESULTS_TITLE)
            
            final_summary_output = gr.Markdown(
                value=UILabels.RESULTS_PLACEHOLDER,
                label=UILabels.RESULTS_LABEL,
                height=500
            )

        # Event handlers (adaptés pour HF Spaces)
        
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
            outputs=[audio_input, audio_section, video_section, input_mode, gr.State("french")]
        )

        # Fonctions de présélection des sections (identiques à l'original)
        def preset_action():
            return (True, True, True, True, True, False, False, False, False)
        
        def preset_info():
            return (True, False, False, False, False, True, True, True, True)
        
        def preset_complet():
            return (True, True, True, True, True, True, True, True, True)
        
        # Gestion de l'analyse directe (adaptée pour Transformers uniquement)
        def handle_analysis_direct(
            audio_file, hf_token, language, local_model, start_trim, end_trim,
            s_resume, s_discussions, s_plan_action, s_decisions, s_prochaines_etapes,
            s_sujets_principaux, s_points_importants, s_questions, s_elements_suivi
        ):
            # Mode Transformers uniquement (pré-quantisé 8-bit)
            transcription_mode = f"Transformers ({local_model} 8-bit)"
            model_key = local_model
            
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
            
            # Appeler la fonction d'analyse directe (chunk duration automatique)
            _, summary = handle_direct_transcription(
                audio_file, hf_token, language, transcription_mode,
                model_key, selected_sections, start_trim, end_trim
            )
            return summary

        # Événements de présélection (identiques à l'original)
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

        # Analyse principale (adaptée pour HF Spaces)
        btn_direct_transcribe.click(
            fn=handle_analysis_direct,
            inputs=[
                audio_input,
                gr.State(value=hf_token),
                gr.State("french"),
                local_model_choice,
                start_trim_input,
                end_trim_input,
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


        # Footer (identique à l'original)
        with gr.Row():
            gr.Markdown(
                """
                ---
                **MeetingNotes** | Powered by [Voxtral](https://mistral.ai/) |
                🚀 Intelligent meeting analysis | 💾 HF Spaces with Zero GPU
                """,
                elem_classes="footer-info"
            )

    # Retourner demo avec le thème et le CSS pour Gradio 6
    return demo, custom_glass_theme, custom_css