"""
Gestionnaires d'événements pour l'interface Gradio.

Ce module contient les fonctions de callback pour les interactions
utilisateur avec l'interface web. Inclut la gestion optimisée de la mémoire
pour les modèles Voxtral.
"""

import gradio as gr
import tempfile
import os
from ..core import (
    process_file_direct_voxtral,
    on_audio_instruct_summary,
    on_audio_direct_analysis_api
)





def handle_direct_transcription(
    file,
    hf_token, 
    language,
    transcription_mode,
    mistral_api_key,
    meeting_type,
    reference_speakers_data=None,
    start_trim=0,
    end_trim=0,
    chunk_duration_minutes=15
):
    """
    Gère la transcription directe sans diarisation.
    
    Args:
        file: Fichier uploadé
        hf_token (str): Token HF
        language (str): Langue
        transcription_mode (str): Mode de transcription
        mistral_api_key (str): Clé API Mistral
        meeting_type (str): Type de réunion
        reference_speakers_data: Inutilisé (conservé pour compatibilité)
        start_trim (float): Secondes à enlever au début
        end_trim (float): Secondes à enlever à la fin
        chunk_duration_minutes (int): Durée des chunks en minutes
    
    Returns:
        tuple: (transcription, final_summary)
    """
    try:
        # Convertir le type de réunion
        meeting_type_code = "action" if "action" in meeting_type.lower() else "information"
        
        # Analyser le mode de transcription pour extraire le modèle
        is_api_mode = "API" in transcription_mode
        
        # Extraire le modèle du mode de transcription
        if is_api_mode:
            # Format: "API (voxtral-mini-latest)" ou "API (voxtral-small-latest)"
            model_name = transcription_mode.replace("API (", "").replace(")", "")
        else:
            # Format: "Local (Voxtral-Mini-3B-2507)" ou "Local (Voxtral-Small-24B-2507)"
            model_name = transcription_mode.replace("Local (", "").replace(")", "")
        
        # Traitement du fichier avec les paramètres de trim
        wav_path = process_file_direct_voxtral(file, hf_token, start_trim, end_trim)
        
        if not wav_path:
            return "", ""
        
        if is_api_mode:
            # Vérifier que la clé API est fournie
            if not mistral_api_key or not mistral_api_key.strip():
                return "❌ Clé API Mistral requise pour le mode API.", ""
            
            # Mode API Mistral - analyse directe avec modèle choisi
            results = on_audio_direct_analysis_api(
                wav_path=wav_path,
                mistral_api_key=mistral_api_key,
                model_name=model_name,
                language=language,
                meeting_type=meeting_type_code,
                chunk_duration_minutes=chunk_duration_minutes,
                reference_speakers_data=None
            )
        else:
            # Mode analyse directe local par chunks avec modèle choisi
            results = on_audio_instruct_summary(
                file=wav_path,
                hf_token=hf_token,
                model_name=model_name,
                language=language,
                meeting_type=meeting_type_code,
                start_trim=0,  # Déjà fait dans process_file_direct_voxtral
                end_trim=0,
                chunk_duration_minutes=chunk_duration_minutes,
                reference_speakers_data=None
            )
        
        analysis_summary = results.get("transcription", "")
        
        # En mode analyse directe, le résumé structuré est déjà complet
        # Pas besoin de génération supplémentaire de résumé final
        return "", analysis_summary  # Transcription vide, résumé structuré dans final_summary
        
    finally:
        # Libération complète de la mémoire après transcription directe
        from ..ai import MemoryManager
        MemoryManager.full_cleanup()


def handle_input_mode_change(mode):
    """
    Gère le changement entre mode Audio et Vidéo.
    
    Args:
        mode (str): Mode sélectionné ("🎵 Audio" ou "🎬 Vidéo")
    
    Returns:
        tuple: Visibilité des sections audio et vidéo
    """
    is_audio_mode = mode == "🎵 Audio"
    return gr.update(visible=is_audio_mode), gr.update(visible=not is_audio_mode)


def extract_audio_from_video(video_file, language):
    """
    Extrait l'audio d'un fichier vidéo et bascule en mode audio.
    
    Args:
        video_file: Fichier vidéo uploadé
        language (str): Langue
    
    Returns:
        tuple: Chemin audio extrait, sections mises à jour, paramètres transférés
    """
    if not video_file:
        return None, gr.update(visible=True), gr.update(visible=False), "🎵 Audio", None
    
    try:
        from ..audio import WavConverter
        
        # Convertir la vidéo en audio
        converter = WavConverter()
        # Gérer le cas où video_file est un chemin ou un objet fichier
        file_path = video_file if isinstance(video_file, str) else video_file.name
        audio_path = converter.convert_to_wav(file_path)
        
        # Basculer en mode audio avec les paramètres transférés
        return (
            audio_path,                    # audio_input
            gr.update(visible=True),       # audio_section
            gr.update(visible=False),      # video_section  
            "🎵 Audio",                    # input_mode
            language                       # language_audio
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction audio : {e}")
        return None, gr.update(), gr.update(), None, None


