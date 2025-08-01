"""
Module d'analyse audio avec l'API Mistral Voxtral.

Ce module utilise l'API Mistral AI pour :
- L'analyse audio multilingue directe via Voxtral
- La génération de résumés structurés depuis l'audio
- L'identification des locuteurs et analyse des prises de parole
- Le traitement des fichiers longs avec découpage automatique

Dépendances:
    - requests: Appels HTTP vers l'API Mistral
    - pydub: Traitement audio pour le découpage
"""

import requests
import tempfile
import json
import time
from pydub import AudioSegment
from typing import List, Dict, Tuple, Optional
import math
import os

from .prompts_config import VoxtralPrompts
from ..utils import format_duration, token_tracker


class VoxtralAPIAnalyzer:
    """
    Analyseur audio intelligent utilisant l'API Mistral Voxtral.
    
    Cette classe gère l'analyse et compréhension audio directe via l'API
    avec support du découpage intelligent pour les fichiers longs.
    Peut identifier les locuteurs et analyser les prises de parole.
    """
    
    def __init__(self, api_key: str, model_name: str = "voxtral-mini-latest"):
        """
        Initialise l'analyseur API Voxtral.
        
        Args:
            api_key (str): Clé API Mistral
            model_name (str): Nom du modèle API à utiliser
        """
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/audio/transcriptions"
        self.model_name = model_name
        self.max_duration_minutes = 10  # Limite API de ~15 min, on prend 10 min pour sécurité
        
        print(f"🔄 Initialisation du transcripteur API Voxtral...")
        print(f"🌐 Mode API avec modèle {self.model_name}")
    
    
    def _get_audio_duration(self, wav_path: str) -> float:
        """
        Obtient la durée d'un fichier audio en minutes.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            
        Returns:
            float: Durée en minutes
        """
        audio = AudioSegment.from_file(wav_path)
        return len(audio) / (1000 * 60)  # Conversion ms -> minutes
    
    def _create_simple_time_chunks(self, wav_path: str) -> List[Tuple[float, float]]:
        """
        Crée des chunks simples basés uniquement sur le temps.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            
        Returns:
            List[Tuple[float, float]]: Liste de tuples (start_time, end_time) en secondes
        """
        total_duration = self._get_audio_duration(wav_path) * 60  # En secondes
        max_chunk_seconds = self.max_duration_minutes * 60
        
        if total_duration <= max_chunk_seconds:
            return [(0, total_duration)]
        
        chunks = []
        current_start = 0
        
        while current_start < total_duration:
            chunk_end = min(current_start + max_chunk_seconds, total_duration)
            chunks.append((current_start, chunk_end))
            current_start = chunk_end
        
        return chunks
    
    def _create_smart_chunks(self, wav_path: str, segments: List[Dict]) -> List[Tuple[float, float]]:
        """
        Crée des chunks intelligents en évitant de couper au milieu des paroles.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            segments (List[Dict]): Segments de diarisation
            
        Returns:
            List[Tuple[float, float]]: Liste de tuples (start_time, end_time) en secondes
        """
        total_duration = self._get_audio_duration(wav_path) * 60  # En secondes
        max_chunk_seconds = self.max_duration_minutes * 60
        
        if total_duration <= max_chunk_seconds:
            return [(0, total_duration)]
        
        chunks = []
        current_start = 0
        
        while current_start < total_duration:
            # Point cible pour la fin du chunk
            target_end = min(current_start + max_chunk_seconds, total_duration)
            
            # Trouver le meilleur point de coupe (fin d'un segment de parole)
            best_cut_point = target_end
            
            # Chercher dans les 2 dernières minutes du chunk un bon point de coupe
            search_start = max(current_start, target_end - 120)  # 2 minutes avant
            
            for segment in segments:
                seg_end = segment["end"]
                # Si la fin du segment est dans notre zone de recherche
                if search_start <= seg_end <= target_end:
                    # Garder le point le plus proche de notre cible
                    if abs(seg_end - target_end) < abs(best_cut_point - target_end):
                        best_cut_point = seg_end
            
            chunks.append((current_start, best_cut_point))
            current_start = best_cut_point
            
            # Éviter les boucles infinies
            if len(chunks) > 50:  # Sécurité pour les très longs fichiers
                break
        
        return chunks
    
    def _extract_audio_chunk(self, wav_path: str, start_time: float, end_time: float) -> str:
        """
        Extrait un chunk audio entre deux timestamps.
        
        Args:
            wav_path (str): Chemin vers le fichier audio source
            start_time (float): Début en secondes
            end_time (float): Fin en secondes
            
        Returns:
            str: Chemin vers le fichier chunk temporaire
        """
        audio = AudioSegment.from_file(wav_path)
        
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        chunk = audio[start_ms:end_ms]
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            chunk_path = tmp_chunk.name
        
        chunk.export(chunk_path, format="wav")
        return chunk_path
    
    def _call_transcription_api(self, audio_path: str, language: str, max_retries: int = 3) -> str:
        """
        Appelle l'API Mistral pour la transcription d'un fichier audio avec retry.
        
        Args:
            audio_path (str): Chemin vers le fichier audio
            language (str): Langue attendue
            max_retries (int): Nombre maximum de tentatives
            
        Returns:
            str: Transcription retournée par l'API
        """
        headers = {
            "x-api-key": self.api_key
        }
        
        # Convertir le code langue si nécessaire
        lang_code = "fr" if language == "french" else "en"
        
        data = {
            "model": self.model_name,
            "language": lang_code
        }
        
        for attempt in range(max_retries):
            files = None
            try:
                files = {
                    "file": open(audio_path, "rb")
                }
                
                print(f"🔄 Tentative {attempt + 1}/{max_retries} pour l'API...")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    transcription = result.get("text", "")
                    print(f"✅ API réussie à la tentative {attempt + 1}")
                    return transcription
                else:
                    error_msg = f"❌ Erreur API: {response.status_code} - {response.text}"
                    print(error_msg)
                    
                    # Ne pas retry pour les erreurs 4xx (erreurs client)
                    if 400 <= response.status_code < 500:
                        return f"❌ Erreur API Mistral: {response.status_code}"
                    
                    # Retry pour les erreurs 5xx (erreurs serveur)
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Backoff exponentiel: 1s, 2s, 4s
                        print(f"⏳ Attente {wait_time}s avant nouvelle tentative...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"❌ Erreur API Mistral après {max_retries} tentatives: {response.status_code}"
                        
            except requests.exceptions.Timeout as e:
                error_msg = f"❌ Timeout API (tentative {attempt + 1}): {e}"
                print(error_msg)
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Backoff exponentiel
                    print(f"⏳ Attente {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"❌ Timeout API après {max_retries} tentatives"
                    
            except Exception as e:
                error_msg = f"❌ Erreur lors de l'appel API (tentative {attempt + 1}): {e}"
                print(error_msg)
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⏳ Attente {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"❌ Erreur API après {max_retries} tentatives: {str(e)}"
                    
            finally:
                # Toujours fermer le fichier
                if files and "file" in files:
                    try:
                        files["file"].close()
                    except:
                        pass
        
        return f"❌ Échec après {max_retries} tentatives"
    
    def transcribe_and_understand(
        self, 
        wav_path: str, 
        segments: List[Dict] = None,
        language: str = "french",
        include_summary: bool = True,
        meeting_type: str = "information"
    ) -> Dict[str, str]:
        """
        Transcrit l'audio avec l'API Voxtral.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            segments (List[Dict], optional): Segments de diarisation (None pour mode simple)
            language (str): Langue attendue
            include_summary (bool): Inclure un résumé dans la réponse
            meeting_type (str): Type de réunion (pour compatibilité, non utilisé en API)
            
        Returns:
            Dict[str, str]: Résultats avec 'transcription'
        """
        duration = self._get_audio_duration(wav_path)
        print(f"🎵 Durée audio: {duration:.1f} minutes")
        
        # Choix du mode de découpage
        if segments is not None:
            # Mode intelligent avec diarisation
            chunks = self._create_smart_chunks(wav_path, segments)
            print(f"📦 Division intelligente en {len(chunks)} chunks (avec diarisation)")
        else:
            # Mode simple sans diarisation
            chunks = self._create_simple_time_chunks(wav_path)
            print(f"📦 Division simple en {len(chunks)} chunks (par temps)")
        
        # Liste pour stocker les résultats de transcription
        transcription_parts = []
        
        for i, (start_time, end_time) in enumerate(chunks):
            print(f"🎯 Traitement du chunk {i+1}/{len(chunks)} ({start_time/60:.1f}-{end_time/60:.1f}min)")
            
            # Extraire le chunk audio
            chunk_path = self._extract_audio_chunk(wav_path, start_time, end_time)
            
            try:
                # Appel API pour transcription
                transcription = self._call_transcription_api(chunk_path, language)
                
                if transcription and not transcription.startswith("❌"):
                    transcription_parts.append(transcription.strip())
                    print(f"✅ Chunk {i+1} traité")
                else:
                    print(f"⚠️ Chunk {i+1} a échoué: {transcription}")
                    transcription_parts.append(f"[Erreur chunk {i+1}: {transcription}]")
                
            finally:
                # Nettoyer le fichier temporaire
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
        
        # Assembler les résultats
        result = {
            "transcription": "\n\n".join(transcription_parts)
        }
        
        return result
    
    def _adjust_diarization_timestamps(self, reference_speakers_data: str, start_offset_seconds: float, chunk_duration_seconds: float) -> str:
        """
        Ajuste les timestamps de diarisation selon l'offset du segment audio.
        Ne garde que les segments qui sont réellement dans ce chunk.
        
        Args:
            reference_speakers_data (str): Données de diarisation avec balises
            start_offset_seconds (float): Décalage en secondes du début du segment
            chunk_duration_seconds (float): Durée du chunk en secondes
            
        Returns:
            str: Diarisation ajustée avec les nouveaux timestamps
        """
        if not reference_speakers_data or not reference_speakers_data.strip():
            return reference_speakers_data
        
        adjusted_lines = []
        
        for line in reference_speakers_data.split('\n'):
            if '<début>' in line and '<fin>' in line:
                # Extraire les timestamps actuels
                try:
                    start_tag_start = line.find('<début>') + len('<début>')
                    start_tag_end = line.find('</début>')
                    end_tag_start = line.find('<fin>') + len('<fin>')
                    end_tag_end = line.find('</fin>')
                    
                    original_start = float(line[start_tag_start:start_tag_end])
                    original_end = float(line[end_tag_start:end_tag_end])
                    
                    # Vérifier si le segment a une intersection avec ce chunk
                    chunk_start = start_offset_seconds
                    chunk_end = start_offset_seconds + chunk_duration_seconds
                    
                    # Segment complètement avant ou après ce chunk ? Ignorer
                    if original_end <= chunk_start or original_start >= chunk_end:
                        continue
                    
                    # Calculer l'intersection avec le chunk
                    intersect_start = max(original_start, chunk_start)
                    intersect_end = min(original_end, chunk_end)
                    
                    # Ajuster les timestamps par rapport au début du chunk
                    adjusted_start = intersect_start - start_offset_seconds
                    adjusted_end = intersect_end - start_offset_seconds
                    
                    # S'assurer que les timestamps sont positifs et dans les limites du chunk
                    adjusted_start = max(0, adjusted_start)
                    adjusted_end = min(chunk_duration_seconds, adjusted_end)
                    
                    # Seulement inclure si on a encore une durée significative (>0.1s)
                    if adjusted_end - adjusted_start > 0.1:
                        # Reconstituer la ligne avec les nouveaux timestamps
                        adjusted_line = line[:start_tag_start] + f"{adjusted_start:.3f}" + line[start_tag_end:end_tag_start] + f"{adjusted_end:.3f}" + line[end_tag_end:]
                        adjusted_lines.append(adjusted_line)
                        
                except (ValueError, IndexError) as e:
                    # En cas d'erreur de parsing, garder la ligne originale
                    print(f"⚠️ Erreur ajustement timestamp: {e}")
                    adjusted_lines.append(line)
            else:
                # Ligne sans timestamp, garder telle quelle
                adjusted_lines.append(line)
        
        return '\n'.join(adjusted_lines)
    
    def _synthesize_with_api(self, synthesis_prompt: str) -> str:
        """
        Effectue la synthèse finale via l'API Mistral en mode texte.
        
        Args:
            synthesis_prompt (str): Prompt de synthèse
            
        Returns:
            str: Résumé synthétisé
        """
        try:
            from mistralai import Mistral
            
            client = Mistral(api_key=self.api_key)
            
            # Utiliser le modèle spécifié pour la synthèse
            response = client.chat.complete(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": synthesis_prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Erreur API synthèse: {e}")
            raise e
    
    
    def analyze_audio_chunks_api(
        self, 
        wav_path: str, 
        language: str = "french", 
        selected_sections: list = None,
        chunk_duration_minutes: int = 15,
        reference_speakers_data=None
    ) -> Dict[str, str]:
        """
        Analyse directe de l'audio par chunks via l'API Voxtral.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            language (str): Langue attendue
            meeting_type (str): Type de réunion
            chunk_duration_minutes (int): Durée des chunks en minutes
            reference_speakers_data: Données des locuteurs de référence identifiés
            
        Returns:
            Dict[str, str]: Résultats avec 'transcription' (analyse concaténée)
        """
        import time
        from pydub import AudioSegment
        import tempfile
        import os
        
        # Mesurer le temps total de traitement
        total_start_time = time.time()
        
        # Initialize token tracker for API mode
        token_tracker.reset()
        token_tracker.set_mode("API")
        
        # Obtenir la durée audio
        audio = AudioSegment.from_file(wav_path)
        duration_minutes = len(audio) / (1000 * 60)
        print(f"🎵 Durée audio: {duration_minutes:.1f} minutes")
        
        # Créer des chunks de temps fixes avec overlap de 10 secondes
        chunk_duration = chunk_duration_minutes * 60 * 1000  # Convertir en millisecondes
        overlap_duration = 10 * 1000  # 10 secondes d'overlap en millisecondes
        chunks = []
        current_time = 0
        
        while current_time < len(audio):
            end_time = min(current_time + chunk_duration, len(audio))
            chunks.append((current_time, end_time))
            
            # Pour le prochain chunk, commencer 10 secondes avant la fin du chunk actuel
            # sauf si c'est le dernier chunk
            if end_time < len(audio):
                current_time = max(0, end_time - overlap_duration)
            else:
                break
        
        print(f"📦 Division en {len(chunks)} chunks de {chunk_duration_minutes} minutes")
        
        # Liste pour stocker les analyses de chaque chunk
        chunk_analyses = []
        
        for i, (start_ms, end_ms) in enumerate(chunks):
            print(f"🎯 Analyse du chunk {i+1}/{len(chunks)} ({start_ms/60000:.1f}-{end_ms/60000:.1f}min)")
            
            # Mesurer le temps de traitement du chunk
            chunk_start_time = time.time()
            
            # Extraire le chunk audio
            chunk = audio[start_ms:end_ms]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
                chunk_path = tmp_chunk.name
            
            chunk.export(chunk_path, format="wav")
            
            try:
                # Ajuster la diarisation selon l'offset temporel du chunk
                adjusted_speaker_context = ""
                if reference_speakers_data:
                    start_offset_seconds = start_ms / 1000.0
                    chunk_duration_seconds = (end_ms - start_ms) / 1000.0
                    adjusted_speaker_context = self._adjust_diarization_timestamps(reference_speakers_data, start_offset_seconds, chunk_duration_seconds)
                
                # Créer le prompt de résumé structuré direct depuis la config centralisée
                sections_list = selected_sections if selected_sections else ["resume_executif"]
                
                # Indiquer que c'est un segment d'un audio plus long seulement s'il y a plusieurs chunks
                chunk_info = f"SEGMENT {i+1}/{len(chunks)} ({start_ms/60000:.1f}-{end_ms/60000:.1f}min)" if len(chunks) > 1 else None
                prompt = VoxtralPrompts.get_meeting_summary_prompt(sections_list, adjusted_speaker_context, chunk_info, None)
                
                
                # Utiliser l'API Chat avec audio
                analysis = self._analyze_audio_chunk_api(chunk_path, prompt)
                
                
                if analysis and not analysis.startswith("❌"):
                    chunk_analyses.append(f"## Segment {i+1} ({start_ms/60000:.1f}-{end_ms/60000:.1f}min)\n\n{analysis}")
                    
                    # Plus de contexte cumulé
                        
                else:
                    chunk_analyses.append(f"## Segment {i+1}\n\n❌ Erreur lors de l'analyse de ce segment: {analysis if analysis else 'Résultat vide'}")
                
                # Calculer et afficher le temps de traitement
                chunk_duration_time = time.time() - chunk_start_time
                print(f"✅ Chunk {i+1} analysé en {format_duration(chunk_duration_time)}")
                
            except Exception as e:
                print(f"❌ Erreur chunk {i+1}: {e}")
                chunk_analyses.append(f"## Segment {i+1}\n\n❌ Erreur de traitement: {str(e)}")
            finally:
                # Nettoyer le fichier temporaire
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        # Traitement final selon le nombre de chunks
        if chunk_analyses:
            if len(chunk_analyses) == 1:
                # Un seul chunk, pas besoin de synthèse
                final_analysis = chunk_analyses[0]
                print(f"✅ Analyse directe API terminée")
            else:
                # Plusieurs chunks : synthèse finale en mode texte
                print(f"🔄 Synthèse finale API en mode texte des {len(chunk_analyses)} segments...")
                combined_content = "\n\n".join(chunk_analyses)
                final_analysis = self._synthesize_chunks_final_api(combined_content, selected_sections)
                print(f"✅ Analyse API avec synthèse finale terminée avec {len(chunk_analyses)} segments")
        else:
            final_analysis = "Aucune analyse disponible."
        
        # Calculer et afficher le temps total
        total_duration = time.time() - total_start_time
        print(f"⏱️ Analyse directe API totale en {format_duration(total_duration)} pour {duration_minutes:.1f}min d'audio")
        
        # Print token usage summary
        token_tracker.print_summary()
        
        return {"transcription": final_analysis}
    
    def _synthesize_chunks_final_api(self, combined_content: str, selected_sections: list) -> str:
        """
        Fait une synthèse finale en mode texte via l'API de tous les segments analysés.
        
        Args:
            combined_content (str): Contenu combiné de tous les segments
            selected_sections (list): Sections sélectionnées pour le résumé
            
        Returns:
            str: Synthèse finale structurée
        """
        try:
            # Créer le prompt pour la synthèse finale
            sections_text = ""
            if selected_sections:
                from .prompts_config import VoxtralPrompts
                for section_key in selected_sections:
                    if section_key in VoxtralPrompts.AVAILABLE_SECTIONS:
                        section = VoxtralPrompts.AVAILABLE_SECTIONS[section_key]
                        sections_text += f"\n{section['title']}\n{section['description']}\n"
            
            synthesis_prompt = f"""Voici les analyses détaillées de plusieurs segments d'une réunion :

{combined_content}

INSTRUCTION CRITIQUE - LANGUE DE RÉPONSE :
- DÉTECTE la langue utilisée dans les segments ci-dessus
- RÉPONDS OBLIGATOIREMENT dans cette même langue détectée
- Si les segments sont en français → réponds en français
- Si les segments sont en anglais → réponds en anglais

Synthétise maintenant ces analyses en un résumé global cohérent et structuré selon les sections demandées :{sections_text}

Fournis une synthèse unifiée qui combine et résume les informations de tous les segments de manière cohérente."""

            # Générer la synthèse avec l'API Mistral en mode texte
            from mistralai import Mistral
            
            client = Mistral(api_key=self.api_key)
            
            response = client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=4000,
                temperature=0.1
            )
            
            final_synthesis = response.choices[0].message.content.strip()
            
            # Track synthesis tokens if usage information is available
            if hasattr(response, 'usage') and response.usage:
                synthesis_input_tokens = response.usage.prompt_tokens or 0
                synthesis_output_tokens = response.usage.completion_tokens or 0
                token_tracker.add_synthesis_tokens(synthesis_input_tokens, synthesis_output_tokens)
            
            return f"# Résumé Global de Réunion\n\n{final_synthesis}\n\n---\n\n## Détails par Segment\n\n{combined_content}"
            
        except Exception as e:
            print(f"❌ Erreur lors de la synthèse finale API: {e}")
            return f"# Résumé de Réunion\n\n⚠️ Erreur lors de la synthèse finale API: {str(e)}\n\n## Analyses par Segment\n\n{combined_content}"
    
    def _analyze_audio_chunk_api(self, audio_path: str, prompt: str) -> str:
        """
        Analyse un chunk audio directement via l'API Chat de Mistral avec audio.
        
        Args:
            audio_path (str): Chemin vers le fichier audio
            prompt (str): Prompt d'analyse
            
        Returns:
            str: Résultat de l'analyse
        """
        import base64
        
        # Encoder le fichier audio en base64
        try:
            with open(audio_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        except Exception as e:
            return f"❌ Erreur lecture fichier audio: {str(e)}"
        
        # Utiliser l'API Chat avec audio directement
        chat_url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": audio_base64
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 16000,  # 16k tokens pour éviter les timeouts API
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                chat_url,
                headers=headers,
                json=data,
                timeout=120  # Timeout plus long pour l'audio
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                
                # Afficher les stats API
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                token_tracker.add_chunk_tokens(input_tokens, output_tokens)
                
                return analysis.strip()
            else:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = error_data.get("error", {}).get("message", response.text)
                except:
                    error_detail = response.text
                return f"❌ Erreur API Chat avec audio: {response.status_code} - {error_detail}"
                
        except Exception as e:
            return f"❌ Erreur appel API Chat avec audio: {str(e)}"

    def _limit_audio_to_ten_minutes(self, wav_path: str) -> str:
        """
        Limite l'audio aux 10 premières minutes pour l'analyse des locuteurs.
        
        Args:
            wav_path (str): Chemin vers le fichier audio source
            
        Returns:
            str: Chemin vers le fichier audio limité à 10 minutes
        """
        from pydub import AudioSegment
        import tempfile
        import os
        
        try:
            # Charger l'audio
            audio = AudioSegment.from_file(wav_path)
            
            # Limiter à 10 minutes (600 secondes = 600 000 ms)
            max_duration_ms = 10 * 60 * 1000  # 10 minutes en millisecondes
            
            if len(audio) <= max_duration_ms:
                # L'audio fait déjà 10 minutes ou moins, retourner le fichier original
                print(f"🎵 Audio fait {len(audio)//1000}s, pas de découpage nécessaire")
                return wav_path
            
            # Découper les 10 premières minutes
            limited_audio = audio[:max_duration_ms]
            
            # Créer un fichier temporaire pour l'audio limité
            with tempfile.NamedTemporaryFile(suffix="_limited_10min.wav", delete=False) as tmp_file:
                limited_path = tmp_file.name
            
            # Exporter l'audio limité
            limited_audio.export(limited_path, format="wav")
            print(f"🎵 Audio limité aux 10 premières minutes : {limited_path}")
            
            return limited_path
            
        except Exception as e:
            print(f"❌ Erreur lors de la limitation audio: {e}")
            return wav_path  # Fallback sur le fichier original

    def analyze_speakers(
        self, 
        wav_path: str, 
        num_speakers: int = None
    ) -> str:
        """
        Analyse les locuteurs d'un fichier audio via l'API et génère un tableau des prises de parole.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            num_speakers (int, optional): Nombre de locuteurs attendu
            
        Returns:
            str: Tableau markdown des prises de parole par locuteur
        """
        print(f"🌐 Analyse des locuteurs avec l'API Voxtral...")
        
        # Limiter l'audio aux 10 premières minutes
        limited_wav_path = self._limit_audio_to_ten_minutes(wav_path)
        
        # Mesurer le temps de traitement
        start_time = time.time()
        
        try:
            # Cette fonctionnalité a été supprimée
            return "⚠️ Fonctionnalité d'identification des locuteurs supprimée"

            # Utiliser l'API Chat avec audio directement
            analysis_result = self._analyze_audio_chunk_api(limited_wav_path, prompt_text)
            
            # Parser le tableau et créer les extraits audio
            if analysis_result and not analysis_result.startswith("❌"):
                enriched_result = self._parse_speakers_table_and_create_snippets(analysis_result, limited_wav_path)
            else:
                enriched_result = analysis_result
            
            # Calculer et afficher le temps de traitement
            duration = time.time() - start_time
            print(f"✅ Analyse des locuteurs API terminée en {format_duration(duration)}")
            
            return enriched_result
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse des locuteurs API: {e}")
            return f"❌ Erreur lors de l'analyse des locuteurs API: {str(e)}"
        
        finally:
            # Nettoyer le fichier temporaire limité s'il a été créé
            if limited_wav_path != wav_path and os.path.exists(limited_wav_path):
                try:
                    os.unlink(limited_wav_path)
                    print(f"🧹 Fichier temporaire limité supprimé: {limited_wav_path}")
                except:
                    pass

    def _parse_speakers_table_and_create_snippets(self, analysis_result: str, wav_path: str) -> str:
        """
        Parse le tableau d'analyse des locuteurs et crée des extraits audio.
        
        Args:
            analysis_result (str): Résultat avec tableau markdown
            wav_path (str): Chemin vers le fichier audio source
            
        Returns:
            str: Tableau enrichi avec liens vers les extraits audio
        """
        import re
        import tempfile
        import os
        from pydub import AudioSegment
        
        try:
            # Charger l'audio source
            audio = AudioSegment.from_file(wav_path)
            
            # Parser le tableau markdown avec regex
            # Chercher les lignes du tableau qui ne sont pas les en-têtes
            table_pattern = r'\|\s*([^|]+)\s*\|\s*(\d{2}:\d{2})\s*\|\s*(\d{2}:\d{2})\s*\|\s*([^|]+)\s*\|'
            matches = re.findall(table_pattern, analysis_result)
            
            if not matches:
                return analysis_result
            
            # Créer les extraits audio et enrichir le tableau
            enriched_lines = []
            lines = analysis_result.split('\n')
            
            snippet_counter = 0
            for line in lines:
                # Si c'est une ligne de données du tableau
                match = re.search(table_pattern, line)
                if match:
                    speaker, start_time, end_time, content = match.groups()
                    
                    # Convertir les timestamps en millisecondes
                    start_ms = self._time_to_ms(start_time.strip())
                    end_ms = self._time_to_ms(end_time.strip())
                    
                    # Vérifier que les timestamps sont valides
                    if start_ms < len(audio) and end_ms <= len(audio) and start_ms < end_ms:
                        # Extraire le segment audio
                        segment = audio[start_ms:end_ms]
                        
                        # Créer un fichier dans un dossier accessible par Gradio
                        import shutil
                        # Créer un dossier audio_snippets dans le répertoire courant
                        snippets_dir = os.path.join(os.getcwd(), "audio_snippets")
                        os.makedirs(snippets_dir, exist_ok=True)
                        
                        # Créer le fichier temporaire d'abord
                        with tempfile.NamedTemporaryFile(suffix=f"_snippet_{snippet_counter}.wav", delete=False) as tmp_file:
                            temp_path = tmp_file.name
                        
                        # Puis le copier vers le dossier accessible
                        snippet_filename = f"snippet_{snippet_counter}_{int(time.time())}.wav"
                        snippet_path = os.path.join(snippets_dir, snippet_filename)
                        
                        segment.export(temp_path, format="wav")
                        # Copier vers le dossier accessible
                        shutil.move(temp_path, snippet_path)
                        snippet_counter += 1
                        
                        # Enrichir la ligne avec le lien audio
                        audio_link = f'<audio controls><source src="{snippet_path}" type="audio/wav"></audio>'
                        enriched_line = f"| {speaker.strip()} | {start_time.strip()} | {end_time.strip()} | {content.strip()} | {audio_link} |"
                        enriched_lines.append(enriched_line)
                        
                    else:
                        pass  # Ignorer silencieusement les timestamps invalides
                        enriched_lines.append(line)
                elif '|' in line and ('Locuteur' in line or 'Speaker' in line) and ('Début' in line or 'Start' in line):
                    # C'est la ligne d'en-tête, ajouter colonne Audio
                    if 'Audio' not in line:
                        enriched_line = line.rstrip(' |') + ' | Audio |'
                        enriched_lines.append(enriched_line)
                    else:
                        enriched_lines.append(line)
                elif '|' in line and all(char in '|-: ' for char in line.strip()):
                    # C'est la ligne de séparation, ajouter colonne pour Audio
                    enriched_line = line.rstrip(' |') + ' |-------|'
                    enriched_lines.append(enriched_line)
                else:
                    # Autres lignes (texte, titre, etc.)
                    enriched_lines.append(line)
            
            return '\n'.join(enriched_lines)
            
        except Exception as e:
            print(f"❌ Erreur lors du parsing et création d'extraits: {e}")
            return analysis_result

    def _time_to_ms(self, time_str: str) -> int:
        """Convertit un timestamp mm:ss en millisecondes."""
        try:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            return (minutes * 60 + seconds) * 1000
        except:
            return 0

    def cleanup_model(self):
        """
        Méthode de compatibilité pour le nettoyage.
        
        Pour l'API, il n'y a pas de modèle local à nettoyer,
        mais cette méthode est nécessaire pour la compatibilité
        avec l'interface existante.
        """
        print("🧹 Nettoyage API (aucune action nécessaire)")
        # Pas de nettoyage spécial nécessaire pour l'API