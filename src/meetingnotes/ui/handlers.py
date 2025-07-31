"""
Gestionnaires d'événements pour l'interface Gradio.

Ce module contient les fonctions de callback pour les interactions
utilisateur avec l'interface web. Inclut la gestion optimisée de la mémoire
pour les modèles Voxtral.
"""

import gradio as gr
import tempfile
import os

# Variables globales pour stocker les segments de référence actuels et les renommages
current_reference_segments = []
current_speaker_renames = {}  # Dict: SPEAKER_XX -> Nom_personnalisé
current_selected_speaker = None
current_rttm_output = ""  # RTTM original pour les modifications
current_diarization_context = ""  # Diarisation formatée pour le contexte de résumé
from ..core import (
    process_file_direct_voxtral,
    on_audio_instruct_summary,
    on_audio_direct_analysis_api,
    on_audio_instruct_summary_mlx
)
from ..ai import PyAnnoteDiarizer


def build_model_name(base_model, precision):
    """
    Construit le nom complet du modèle à partir du modèle de base et de la précision.
    
    Args:
        base_model (str): Modèle de base comme "Voxtral-Mini-3B-2507"
        precision (str): Précision comme "Default", "8bit", "4bit"
        
    Returns:
        str: Nom du modèle réel à utiliser
    """
    # Mapping des modèles avec quantification
    model_mappings = {
        # Voxtral Mini
        ("Voxtral-Mini-3B-2507", "Default"): "mistralai/Voxtral-Mini-3B-2507",
        ("Voxtral-Mini-3B-2507", "8bit"): "mzbac/voxtral-mini-3b-8bit",
        ("Voxtral-Mini-3B-2507", "4bit"): "mzbac/voxtral-mini-3b-4bit-mixed",
        
        # Voxtral Small
        ("Voxtral-Small-24B-2507", "Default"): "mistralai/Voxtral-Small-24B-2507",
        ("Voxtral-Small-24B-2507", "8bit"): "VincentGOURBIN/voxtral-small-8bit",
        ("Voxtral-Small-24B-2507", "4bit"): "VincentGOURBIN/voxtral-small-4bit-mixed",
    }
    
    model_key = (base_model, precision)
    return model_mappings.get(model_key, "mistralai/Voxtral-Mini-3B-2507")





def handle_direct_transcription(
    file,
    hf_token, 
    language,
    transcription_mode,
    mistral_api_key,
    selected_sections,
    reference_speakers_data=None,
    start_trim=0,
    end_trim=0,
    chunk_duration_minutes=15
):
    """
    Gère la transcription directe avec sections personnalisables.
    
    Args:
        file: Fichier uploadé
        hf_token (str): Token HF
        language (str): Langue
        transcription_mode (str): Mode de transcription
        mistral_api_key (str): Clé API Mistral
        selected_sections (list): Liste des sections à inclure dans le résumé
        reference_speakers_data: Contexte de diarisation (optionnel)
        start_trim (float): Secondes à enlever au début
        end_trim (float): Secondes à enlever à la fin
        chunk_duration_minutes (int): Durée des chunks en minutes
    
    Returns:
        tuple: (transcription, final_summary)
    """
    try:
        # Valider et normaliser les paramètres de trim
        start_trim = 0 if start_trim is None or start_trim == "" else float(start_trim)
        end_trim = 0 if end_trim is None or end_trim == "" else float(end_trim)
        
        # Analyser le mode de transcription pour extraire le modèle
        is_api_mode = "API" in transcription_mode
        is_mlx_mode = "MLX" in transcription_mode
        
        # Extraire le modèle du mode de transcription
        if is_api_mode:
            # Format: "API (voxtral-mini-latest)" ou "API (voxtral-small-latest)"
            model_name = transcription_mode.replace("API (", "").replace(")", "")
        elif is_mlx_mode:
            # Format: "MLX (Voxtral-Mini-3B-2507 (Default))" etc.
            # Extraire modèle de base et précision
            start_idx = transcription_mode.find("(") + 1
            end_idx = transcription_mode.rfind(")")
            model_and_precision = transcription_mode[start_idx:end_idx]
            
            # Parser "Voxtral-Mini-3B-2507 (Default)" -> base_model, precision
            precision_start = model_and_precision.find(" (") + 2
            precision_end = model_and_precision.rfind(")")
            base_model = model_and_precision[:model_and_precision.find(" (")]
            precision = model_and_precision[precision_start:precision_end]
            
            model_name = build_model_name(base_model, precision)
        else:
            # Format: "Local (Voxtral-Mini-3B-2507 (Default))" etc.
            # Extraire modèle de base et précision
            start_idx = transcription_mode.find("(") + 1
            end_idx = transcription_mode.rfind(")")
            model_and_precision = transcription_mode[start_idx:end_idx]
            
            # Parser "Voxtral-Mini-3B-2507 (Default)" -> base_model, precision
            precision_start = model_and_precision.find(" (") + 2
            precision_end = model_and_precision.rfind(")")
            base_model = model_and_precision[:model_and_precision.find(" (")]
            precision = model_and_precision[precision_start:precision_end]
            
            model_name = build_model_name(base_model, precision)
        
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
                selected_sections=selected_sections,
                chunk_duration_minutes=chunk_duration_minutes,
                reference_speakers_data=reference_speakers_data
            )
        elif is_mlx_mode:
            # Mode MLX - analyse directe avec MLX
            results = on_audio_instruct_summary_mlx(
                file=wav_path,
                model_name=model_name,
                language=language,
                selected_sections=selected_sections,
                chunk_duration_minutes=chunk_duration_minutes,
                reference_speakers_data=reference_speakers_data,
                start_trim=0,  # Déjà fait dans process_file_direct_voxtral
                end_trim=0
            )
        else:
            # Mode analyse directe local par chunks avec modèle choisi
            results = on_audio_instruct_summary(
                file=wav_path,
                hf_token=hf_token,
                model_name=model_name,
                language=language,
                selected_sections=selected_sections,
                start_trim=0,  # Déjà fait dans process_file_direct_voxtral
                end_trim=0,
                chunk_duration_minutes=chunk_duration_minutes,
                reference_speakers_data=reference_speakers_data
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


def handle_diarization(audio_file, hf_token, num_speakers, start_trim=0, end_trim=0):
    """
    Gère la diarisation d'un fichier audio avec pyannote.
    
    Args:
        audio_file: Fichier audio (chemin ou objet Gradio)
        hf_token (str): Token Hugging Face
        num_speakers (int, optional): Nombre de locuteurs attendu
        start_trim (float): Secondes à enlever au début
        end_trim (float): Secondes à enlever à la fin
        
    Returns:
        tuple: (résultat_rttm, boutons_dynamiques, audio_player_vide)
    """
    # Valider et normaliser les paramètres de trim
    start_trim = 0 if start_trim is None or start_trim == "" else float(start_trim)
    end_trim = 0 if end_trim is None or end_trim == "" else float(end_trim)
    
    if not audio_file:
        return "❌ Aucun fichier audio fourni.", gr.update(choices=[]), None
    
    if not hf_token:
        return "❌ Token Hugging Face requis pour la diarisation.", gr.update(choices=[]), None
    
    diarizer = None
    try:
        # Gérer le cas où audio_file est un chemin ou un objet fichier
        file_path = audio_file if isinstance(audio_file, str) else audio_file.name
        
        print(f"🎤 Démarrage de la diarisation: {file_path}")
        
        # Appliquer les trims comme pour Voxtral
        processed_audio_path = process_file_direct_voxtral(file_path, hf_token, start_trim, end_trim)
        if not processed_audio_path:
            return "❌ Erreur lors du traitement du fichier audio.", gr.update(choices=[]), None
        
        print(f"🎵 Audio traité pour diarisation: {processed_audio_path}")
        
        # Initialiser le diariseur
        diarizer = PyAnnoteDiarizer(hf_token)
        
        # Effectuer la diarisation sur le fichier traité
        result = diarizer.diarize_audio(
            audio_path=processed_audio_path,
            num_speakers=num_speakers if num_speakers else None
        )
        
        # Vérifier si on a un tuple (succès) ou un string (erreur)
        if isinstance(result, tuple) and len(result) == 2:
            rttm_result, reference_segments = result
            
            # Créer les boutons dynamiques pour les segments de référence
            speaker_choices = []
            for segment in reference_segments:
                speaker_label = f"🎤 {segment['speaker']} ({segment['duration']:.1f}s)"
                speaker_choices.append(speaker_label)
            
            print(f"📋 Boutons créés: {len(speaker_choices)} locuteurs")
            
            # Stocker les segments et le RTTM pour l'usage ultérieur
            global current_reference_segments, current_rttm_output, current_speaker_renames, current_diarization_context
            current_reference_segments = reference_segments
            current_rttm_output = rttm_result
            current_speaker_renames = {}  # Reset des renommages
            current_diarization_context = convert_rttm_to_tagged_format(rttm_result)  # Format avec balises pour le contexte
            
            # Afficher la section de renommage si on a des locuteurs
            rename_section_visible = len(speaker_choices) > 0
            
            return (
                gr.update(choices=speaker_choices, visible=len(speaker_choices) > 0), 
                None,
                gr.update(visible=rename_section_visible)
            )
        else:
            # Cas d'erreur - result est un string d'erreur
            print(f"❌ Erreur diarisation: {result}")
            return gr.update(choices=[]), None, gr.update(visible=False)
        
    except Exception as e:
        print(f"❌ Erreur lors de la diarisation: {e}")
        return f"❌ Erreur lors de la diarisation: {str(e)}", gr.update(choices=[]), None, gr.update(visible=False)
    
    finally:
        # Libérer les ressources
        if diarizer is not None:
            diarizer.cleanup()


def handle_speaker_selection(speaker_button, current_name_in_field):
    """
    Gère la sélection d'un locuteur pour écouter son segment de référence.
    Sauvegarde automatiquement le nom du locuteur précédent.
    
    Args:
        speaker_button (str): Label du bouton sélectionné
        current_name_in_field (str): Nom actuellement dans le champ de saisie
        
    Returns:
        tuple: (chemin_audio, nom_actuel_du_locuteur)
    """
    global current_reference_segments, current_selected_speaker, current_speaker_renames
    
    # Sauvegarder le nom du locuteur précédent s'il y en avait un
    if current_selected_speaker and current_name_in_field and current_name_in_field.strip():
        old_name = current_name_in_field.strip()
        if old_name != current_selected_speaker:  # Seulement si différent de l'ID original
            current_speaker_renames[current_selected_speaker] = old_name
            print(f"💾 Sauvegarde automatique: {current_selected_speaker} → {old_name}")
    
    if not speaker_button or not current_reference_segments:
        return None, ""
    
    try:
        # Extraire le nom du locuteur du label du bouton (format: "🎤 SPEAKER_XX (5.2s)")
        speaker_name = speaker_button.split(" ")[1]  # Récupère "SPEAKER_XX"
        current_selected_speaker = speaker_name
        
        print(f"🔍 Recherche du segment pour: {speaker_name}")
        
        # Chercher le segment correspondant
        for segment in current_reference_segments:
            if segment['speaker'] == speaker_name:
                print(f"✅ Segment trouvé: {segment['audio_path']}")
                
                # Retourner le nom actuel (renommé ou original)
                current_name = current_speaker_renames.get(speaker_name, speaker_name)
                return segment['audio_path'], current_name
        
        print(f"❌ Aucun segment trouvé pour: {speaker_name}")
        return None, ""
        
    except Exception as e:
        print(f"❌ Erreur lors de la sélection du locuteur: {e}")
        return None, ""


def handle_speaker_rename(current_name_in_field):
    """
    Applique tous les renommages à la diarisation RTTM complète.
    
    Args:
        current_name_in_field (str): Nom dans le champ (pour sauvegarder le dernier)
        
    Returns:
        tuple: (rttm_modifié, zone_visible)
    """
    global current_selected_speaker, current_speaker_renames, current_rttm_output, current_diarization_context
    
    # Sauvegarder le nom du locuteur actuellement sélectionné
    if current_selected_speaker and current_name_in_field and current_name_in_field.strip():
        clean_name = current_name_in_field.strip()
        if clean_name != current_selected_speaker:  # Seulement si différent de l'ID original
            current_speaker_renames[current_selected_speaker] = clean_name
            print(f"💾 Sauvegarde finale: {current_selected_speaker} → {clean_name}")
    
    if not current_rttm_output or not current_speaker_renames:
        return "", gr.update(visible=False)
    
    # Mettre à jour le contexte de diarisation avec les nouveaux noms (format avec balises)
    current_diarization_context = apply_renames_to_rttm(current_rttm_output, current_speaker_renames)
    
    # Générer le résumé lisible des locuteurs pour l'affichage utilisateur
    speakers_summary = generate_speakers_summary(current_rttm_output, current_speaker_renames)
    
    print(f"✅ Locuteurs renommés: {len(current_speaker_renames)} modifications")
    print(f"📋 Contexte de diarisation mis à jour pour les futurs résumés")
    
    return speakers_summary, gr.update(visible=True)


def apply_renames_to_rttm(original_rttm: str, renames: dict) -> str:
    """
    Applique les renommages de locuteurs au RTTM complet.
    
    Args:
        original_rttm (str): RTTM original
        renames (dict): Dict des renommages SPEAKER_XX -> Nom
        
    Returns:
        str: RTTM modifié avec les nouveaux noms
    """
    if not original_rttm or not renames:
        return original_rttm
    
    modified_lines = []
    
    for line in original_rttm.split('\n'):
        if line.strip() and line.startswith('SPEAKER'):
            # Parser la ligne RTTM: SPEAKER file 1 start duration <NA> <NA> speaker_id <NA> <NA>
            parts = line.split()
            if len(parts) >= 8:
                speaker_id = parts[7]  # L'ID du locuteur est à l'index 7
                
                # Remplacer par le nom personnalisé si disponible
                if speaker_id in renames:
                    custom_name = renames[speaker_id]
                    start_time = parts[3]
                    duration = parts[4]
                    
                    # Format avec balises: "<locuteur>Jean</locuteur> <début>358.270</début> <fin>360.666</fin>"
                    end_time = float(start_time) + float(duration)
                    modified_line = f"<locuteur>{custom_name}</locuteur> <début>{start_time}</début> <fin>{end_time:.3f}</fin>"
                    modified_lines.append(modified_line)
                else:
                    # Format avec balises même sans renommage
                    start_time = parts[3]
                    duration = parts[4]
                    end_time = float(start_time) + float(duration)
                    modified_line = f"<locuteur>{speaker_id}</locuteur> <début>{start_time}</début> <fin>{end_time:.3f}</fin>"
                    modified_lines.append(modified_line)
            else:
                # Ligne malformée, garder telle quelle
                modified_lines.append(line)
        else:
            # Ligne non-SPEAKER, garder telle quelle
            modified_lines.append(line)
    
    return '\n'.join(modified_lines)


def convert_rttm_to_tagged_format(rttm_output: str) -> str:
    """
    Convertit un RTTM brut au format avec balises pour le contexte de diarisation.
    
    Args:
        rttm_output (str): RTTM brut de pyannote
        
    Returns:
        str: RTTM formaté avec balises
    """
    if not rttm_output:
        return ""
    
    tagged_lines = []
    
    for line in rttm_output.split('\n'):
        if line.strip() and line.startswith('SPEAKER'):
            # Parser la ligne RTTM: SPEAKER file 1 start duration <NA> <NA> speaker_id <NA> <NA>
            parts = line.split()
            if len(parts) >= 8:
                speaker_id = parts[7]  # L'ID du locuteur est à l'index 7
                start_time = parts[3]
                duration = parts[4]
                end_time = float(start_time) + float(duration)
                
                # Format avec balises
                tagged_line = f"<locuteur>{speaker_id}</locuteur> <début>{start_time}</début> <fin>{end_time:.3f}</fin>"
                tagged_lines.append(tagged_line)
    
    return '\n'.join(tagged_lines)


def generate_speakers_summary(original_rttm: str, renames: dict) -> str:
    """
    Génère un résumé lisible des locuteurs identifiés.
    
    Args:
        original_rttm (str): RTTM original
        renames (dict): Dict des renommages SPEAKER_XX -> Nom
        
    Returns:
        str: Résumé lisible des locuteurs
    """
    if not original_rttm:
        return ""
    
    # Compter les segments par locuteur
    speaker_segments = {}
    total_duration_by_speaker = {}
    
    for line in original_rttm.split('\n'):
        if line.strip() and line.startswith('SPEAKER'):
            parts = line.split()
            if len(parts) >= 8:
                speaker_id = parts[7]
                duration = float(parts[4])
                
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = 0
                    total_duration_by_speaker[speaker_id] = 0.0
                
                speaker_segments[speaker_id] += 1
                total_duration_by_speaker[speaker_id] += duration
    
    # Générer le résumé
    summary_lines = []
    for speaker_id in sorted(speaker_segments.keys()):
        display_name = renames.get(speaker_id, speaker_id)
        segment_count = speaker_segments[speaker_id]
        total_time = total_duration_by_speaker[speaker_id]
        
        # Formatage du temps (minutes:secondes)
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        time_str = f"{minutes}:{seconds:02d}"
        
        summary_lines.append(f"👤 {display_name}: {segment_count} segments, {time_str} de parole")
    
    if summary_lines:
        return "\n".join(summary_lines)
    else:
        return "Aucun locuteur identifié"




