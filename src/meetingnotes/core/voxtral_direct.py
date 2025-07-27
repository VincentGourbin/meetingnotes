"""
Module pour transcription directe avec Voxtral sans diarisation préalable.

Ce module fournit des fonctions pour traiter directement l'audio
avec Voxtral sans passer par l'étape de diarisation.
"""

import os
import tempfile
from pydub import AudioSegment
from ..audio import WavConverter, Normalizer
from ..ai import VoxtralAnalyzer, MemoryManager, auto_cleanup


def process_file_direct_voxtral(file, hf_token, start_trim=0, end_trim=0):
    """
    Traite un fichier audio/vidéo directement avec Voxtral (sans diarisation).
    
    Args:
        file: Fichier uploadé via Gradio
        hf_token (str): Token Hugging Face
        start_trim (float): Secondes à supprimer au début
        end_trim (float): Secondes à supprimer à la fin
    
    Returns:
        str: Chemin vers le fichier WAV traité
    """
    if file is None:
        return None

    converter = WavConverter()
    normalizer = Normalizer()

    # Convertir en WAV puis normaliser
    # Gérer le cas où file est un chemin de fichier (string) ou un objet fichier
    file_path = file if isinstance(file, str) else file.name
    wav_path = converter.convert_to_wav(file_path)
    wav_path = normalizer.normalize(wav_path)

    # Trim audio si nécessaire
    if start_trim > 0 or end_trim > 0:
        audio = AudioSegment.from_file(wav_path, format="wav")
        duration_ms = len(audio)
        start_ms = start_trim * 1000
        end_ms = duration_ms - (end_trim * 1000)
        if start_ms < end_ms:
            trimmed_audio = audio[start_ms:end_ms]
        else:
            trimmed_audio = audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_trimmed:
            trimmed_path = tmp_trimmed.name
        trimmed_audio.export(trimmed_path, format="wav")
        wav_path = trimmed_path

    return wav_path


def on_transcribe_direct_voxtral(wav_path, hf_token, language, include_summary=True, meeting_type="information"):
    """
    Transcription directe avec Voxtral sans diarisation.
    
    Args:
        wav_path (str): Chemin du fichier audio traité
        hf_token (str): Token Hugging Face
        language (str): Langue ('french' ou 'english')
        include_summary (bool): Inclure un résumé automatique
    
    Returns:
        dict: Résultats avec 'transcription' et optionnellement 'summary'
    """
    if not wav_path:
        return {"transcription": "", "summary": ""} if include_summary else {"transcription": ""}

    analyzer = None
    try:
        # Initialiser Voxtral
        analyzer = VoxtralAnalyzer(hf_token)
        
        # Transcription sans segments de diarisation (mode simple)
        results = analyzer.transcribe_and_understand(
            wav_path=wav_path,
            segments=None,  # Pas de diarisation
            language=language,
            include_summary=include_summary,
            meeting_type=meeting_type
        )
        
        return results
    
    finally:
        # Libérer le modèle de la mémoire
        if analyzer is not None:
            analyzer.cleanup_model()
            
        # Nettoyage final de la mémoire
        MemoryManager.full_cleanup()
        MemoryManager.print_memory_stats("Fin transcription directe")


def on_audio_direct_analysis(
    file,
    hf_token,
    model_name="Voxtral-Mini-3B-2507",
    language="french",
    meeting_type="information",
    start_trim=0,
    end_trim=0,
    chunk_duration_minutes=15,
    reference_speakers_data=None
):
    """
    Analyse directe de chunks audio via l'audio instruct mode.
    
    Args:
        file: Fichier audio (chemin ou objet)
        hf_token (str): Token Hugging Face  
        language (str): Langue de traitement
        meeting_type (str): Type de réunion
        start_trim (float): Secondes à enlever au début
        end_trim (float): Secondes à enlever à la fin
        chunk_duration_minutes (int): Durée des chunks en minutes
        
    Returns:
        Dict[str, str]: Résultats avec analyse directe concaténée
    """
    analyzer = None
    
    try:
        # Traitement du fichier avec trim
        wav_path = process_file_direct_voxtral(file, hf_token, start_trim, end_trim)
        
        if not wav_path:
            return {"transcription": "❌ Erreur lors du traitement du fichier audio."}
        
        # Initialiser Voxtral avec le modèle choisi
        analyzer = VoxtralAnalyzer(hf_token, model_name)
        
        print(f"🎙️ Mode analyse directe avec chunks de {chunk_duration_minutes} minutes")
        
        # Utiliser la méthode d'analyse directe par chunks
        results = analyzer.analyze_audio_chunks(
            wav_path=wav_path,
            language=language,
            meeting_type=meeting_type,
            chunk_duration_minutes=chunk_duration_minutes,
            reference_speakers_data=reference_speakers_data
        )
        
        return results
    
    finally:
        # Libérer le modèle de la mémoire
        if analyzer is not None:
            analyzer.cleanup_model()
            
        # Nettoyage final de la mémoire
        MemoryManager.full_cleanup()
        MemoryManager.print_memory_stats("Fin analyse directe audio")


# Alias pour compatibilité (sera supprimé plus tard)
def on_audio_instruct_summary(
    file,
    hf_token,
    model_name="Voxtral-Mini-3B-2507",
    language="french",
    meeting_type="information",
    start_trim=0,
    end_trim=0,
    chunk_duration_minutes=15,
    reference_speakers_data=None
):
    return on_audio_direct_analysis(
        file, hf_token, model_name, language, meeting_type, 
        start_trim, end_trim, chunk_duration_minutes, reference_speakers_data
    )