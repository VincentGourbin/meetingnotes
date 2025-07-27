"""
Module de traitement direct avec l'API Voxtral de Mistral.

Ce module fournit les fonctions de traitement audio
avec l'API Mistral Voxtral pour la transcription et l'analyse.
"""

from typing import Dict, List, Optional
from ..ai import VoxtralAPIAnalyzer


def on_transcribe_voxtral_api(
    mapping_table,
    segments: List[Dict],
    wav_path: str,
    mistral_api_key: str,
    language: str = "french",
    meeting_type: str = "information"
) -> Dict[str, str]:
    """
    Transcrit l'audio avec l'API Voxtral en mode avec diarisation.
    
    Args:
        mapping_table: Tableau de correspondance des locuteurs
        segments (List[Dict]): Segments de diarisation
        wav_path (str): Chemin vers le fichier audio
        mistral_api_key (str): Clé API Mistral
        language (str): Langue attendue
        meeting_type (str): Type de réunion
        
    Returns:
        Dict[str, str]: Résultats avec 'transcription'
    """
    analyzer = None
    try:
        print("🌐 Démarrage de la transcription avec l'API Voxtral...")
        
        # Créer l'instance de l'analyseur API
        analyzer = VoxtralAPIAnalyzer(mistral_api_key)
        
        # Transcription avec segments de diarisation
        results = analyzer.transcribe_and_understand(
            wav_path=wav_path,
            segments=segments,
            language=language,
            include_summary=False,  # Pas de résumé automatique
            meeting_type=meeting_type
        )
        
        print("✅ Transcription API terminée avec succès")
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de la transcription API: {e}")
        return {"transcription": f"❌ Erreur lors de la transcription API: {str(e)}"}
    
    finally:
        # Pas de nettoyage spécial nécessaire pour l'API
        print("🧹 Nettoyage API terminé")


def on_transcribe_direct_voxtral_api(
    wav_path: str,
    mistral_api_key: str,
    language: str = "french",
    meeting_type: str = "information"
) -> Dict[str, str]:
    """
    Transcrit directement l'audio avec l'API Voxtral (sans diarisation).
    
    Args:
        wav_path (str): Chemin vers le fichier audio
        mistral_api_key (str): Clé API Mistral
        language (str): Langue attendue
        meeting_type (str): Type de réunion
        
    Returns:
        Dict[str, str]: Résultats avec 'transcription'
    """
    analyzer = None
    try:
        print("🌐 Démarrage de la transcription directe avec l'API Voxtral...")
        
        # Créer l'instance de l'analyseur API
        analyzer = VoxtralAPIAnalyzer(mistral_api_key)
        
        # Transcription directe sans segments de diarisation
        results = analyzer.transcribe_and_understand(
            wav_path=wav_path,
            segments=None,  # Pas de diarisation
            language=language,
            include_summary=False,  # Pas de résumé automatique
            meeting_type=meeting_type
        )
        
        print("✅ Transcription API directe terminée avec succès")
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de la transcription API directe: {e}")
        return {"transcription": f"❌ Erreur lors de la transcription API directe: {str(e)}"}
    
    finally:
        # Pas de nettoyage spécial nécessaire pour l'API
        print("🧹 Nettoyage API direct terminé")


def on_audio_direct_analysis_api(
    wav_path: str,
    mistral_api_key: str,
    model_name: str = "voxtral-mini-latest", 
    language: str = "french",
    meeting_type: str = "information",
    chunk_duration_minutes: int = 15,
    reference_speakers_data=None
) -> Dict[str, str]:
    """
    Analyse directe de l'audio par chunks avec l'API Voxtral.
    
    Args:
        wav_path (str): Chemin vers le fichier audio
        mistral_api_key (str): Clé API Mistral
        model_name (str): Nom du modèle API à utiliser
        language (str): Langue attendue
        meeting_type (str): Type de réunion
        chunk_duration_minutes (int): Durée des chunks en minutes
        reference_speakers_data: Données des locuteurs de référence identifiés
        
    Returns:
        Dict[str, str]: Résultats avec 'transcription' (analyse concaténée)
    """
    analyzer = None
    try:
        print("🌐 Démarrage de l'analyse directe avec l'API Voxtral...")
        
        # Créer l'instance de l'analyseur API avec le modèle choisi
        analyzer = VoxtralAPIAnalyzer(mistral_api_key, model_name)
        
        # Analyse directe par chunks avec audio instruct mode
        results = analyzer.analyze_audio_chunks_api(
            wav_path=wav_path,
            language=language,
            meeting_type=meeting_type,
            chunk_duration_minutes=chunk_duration_minutes,
            reference_speakers_data=reference_speakers_data
        )
        
        print("✅ Analyse directe API terminée avec succès")
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse directe API: {e}")
        return {"transcription": f"❌ Erreur lors de l'analyse directe API: {str(e)}"}
    
    finally:
        # Pas de nettoyage spécial nécessaire pour l'API
        print("🧹 Nettoyage analyse directe API terminé")