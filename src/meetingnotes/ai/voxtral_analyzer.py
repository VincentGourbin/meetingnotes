"""
Module d'analyse audio intelligente avec Voxtral.

Ce module utilise le modèle Voxtral de Mistral AI pour :
- L'analyse audio multilingue directe
- La compréhension et l'interprétation du contenu
- La génération de résumés structurés depuis l'audio
- L'identification des locuteurs et analyse des prises de parole

Dépendances:
    - transformers: Framework Hugging Face
    - torch: Framework de deep learning
    - torchaudio: Traitement audio
    - mistral-common: Outils spécifiques Mistral
"""

import torch
import torchaudio
import tempfile
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from pydub import AudioSegment
from typing import List, Dict, Tuple, Optional
import math
import gc
import io
import time
from .memory_manager import MemoryManager, auto_cleanup, cleanup_temp_files
from .prompts_config import VoxtralPrompts
from ..utils import format_duration, token_tracker


class VoxtralAnalyzer:
    """
    Analyseur audio intelligent utilisant Voxtral.
    
    Cette classe gère l'analyse et compréhension audio directe avec support
    du découpage intelligent pour les fichiers longs. Peut identifier les
    locuteurs et analyser les prises de parole.
    """
    
    def __init__(self, hf_token: str, model_name: str = "Voxtral-Mini-3B-2507"):
        """
        Initialise l'analyseur Voxtral.
        
        Args:
            hf_token (str): Token Hugging Face pour accéder aux modèles
            model_name (str): Nom du modèle Voxtral à utiliser
        """
        # Mapper les noms vers les identifiants Hugging Face
        model_mapping = {
            "Voxtral-Mini-3B-2507": "mistralai/Voxtral-Mini-3B-2507",
            "Voxtral-Small-24B-2507": "mistralai/Voxtral-Small-24B-2507"
        }
        
        self.model_name = model_mapping.get(model_name, "mistralai/Voxtral-Mini-3B-2507")
        self.max_duration_minutes = 25  # Augmenté avec max_new_tokens=32k pour moins de chunks
        
        print(f"🔄 Chargement du modèle Voxtral...")
        
        # Chargement du processeur
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=hf_token
        )
        
        # Configuration du device et dtype
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_str = "mps"
            # MPS peut avoir des problèmes avec bfloat16, utilisons float16
            self.dtype = torch.float16
            print(f"🚀 Utilisation de MPS (Apple Silicon) avec float16")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_str = "cuda"
            self.dtype = torch.bfloat16
            print(f"🚀 Utilisation de CUDA GPU avec bfloat16")
        else:
            self.device = torch.device("cpu")
            device_str = "cpu"
            self.dtype = torch.float16  # CPU avec float16 pour économiser la mémoire
            print(f"⚠️  Utilisation du CPU avec float16")
        
        # Chargement direct du modèle sans quantification Quanto
        # (les modèles pré-quantifiés sont gérés automatiquement)
        print("📦 Chargement du modèle pré-quantifié ou standard")
        
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_name,
            token=hf_token,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=device_str
        )
        
        print(f"🚀 Chargement optimisé avec device_map et dtype intelligent")
        
        # Afficher les stats de mémoire après chargement
        MemoryManager.print_memory_stats("Après chargement Voxtral")
        
    
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
        Crée des chunks simples basés uniquement sur le temps (sans diarisation).
        
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
    
    def transcribe_and_understand(
        self, 
        wav_path: str, 
        segments: List[Dict] = None,
        language: str = "french",
        include_summary: bool = True,
        meeting_type: str = "information"
    ) -> Dict[str, str]:
        """
        Transcrit et analyse l'audio avec Voxtral.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            segments (List[Dict], optional): Segments de diarisation (None pour mode simple)
            language (str): Langue attendue
            include_summary (bool): Inclure un résumé dans la réponse
            meeting_type (str): Type de réunion ("information" ou "action")
            
        Returns:
            Dict[str, str]: Résultats avec 'transcription' et optionnellement 'summary'
        """
        # Mesurer le temps total de traitement
        total_start_time = time.time()
        
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
        
        # Listes simples pour stocker les résultats
        transcription_parts = []
        summary_parts = []
        
        for i, (start_time, end_time) in enumerate(chunks):
            print(f"🎯 Traitement du chunk {i+1}/{len(chunks)} ({start_time/60:.1f}-{end_time/60:.1f}min)")
            MemoryManager.print_memory_stats(f"Avant chunk {i+1}")
            
            # Mesurer le temps de traitement du chunk
            chunk_start_time = time.time()
            
            # Extraire le chunk audio
            chunk_path = self._extract_audio_chunk(wav_path, start_time, end_time)
            
            try:
                # Utiliser la méthode spécifique de transcription pour une meilleure efficacité
                if include_summary:
                    # Pour les résumés, utiliser le mode conversation
                    prompt_text = VoxtralPrompts.get_meeting_summary_prompt(meeting_type, "")
                    conversation = [{
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": chunk_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }]
                    inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
                else:
                    # Pour la transcription pure, utiliser la méthode dédiée (plus efficace)
                    language_code = "fr" if language == "french" else "en"
                    inputs = self.processor.apply_transcrition_request(
                        language=language_code, 
                        audio=chunk_path,
                        model_id=self.model_name
                    )
                
                # Placer les inputs sur le device avec le dtype approprié
                inputs = inputs.to(self.device, dtype=self.dtype)
                
                # Génération optimisée avec cache
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=32000,  # Utilisation maximale du contexte Voxtral (32k)
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,  # Optimisation cache
                        output_scores=False  # Pas besoin des scores pour économiser mémoire
                    )
                
                # Décodage simple
                response = self.processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]
                
                # Nettoyage basique
                del inputs, outputs
                
                # Parser la réponse selon le mode
                if include_summary:
                    transcription, summary = self._parse_comprehension_response(response)
                    transcription_parts.append(transcription)
                    summary_parts.append(summary)
                else:
                    transcription_parts.append(response.strip())
                
                # Calculer et afficher le temps de traitement
                chunk_duration = time.time() - chunk_start_time
                print(f"✅ Chunk {i+1} traité en {format_duration(chunk_duration)}")
                
            finally:
                # Nettoyer le fichier temporaire
                import os
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
                
                # Nettoyage de la mémoire avec le gestionnaire + garbage collection forcé
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                MemoryManager.full_cleanup()
                MemoryManager.print_memory_stats(f"Après chunk {i+1}")
        
        # Nettoyage final de la mémoire
        MemoryManager.full_cleanup()
        MemoryManager.print_memory_stats("Fin traitement Voxtral")
        
        # Assembler les résultats simplement
        result = {
            "transcription": "\n\n".join(transcription_parts)
        }
        
        if include_summary and summary_parts:
            result["summary"] = self._merge_summaries(summary_parts)
        
        # Calculer et afficher le temps total
        total_duration = time.time() - total_start_time
        mode_desc = "avec diarisation" if segments is not None else "transcription simple"
        print(f"⏱️ Traitement Voxtral total ({mode_desc}) en {format_duration(total_duration)} pour {duration:.1f}min d'audio")
        
        return result
    
    def _build_transcription_prompt(self, language: str) -> str:
        """Construit le prompt pour la transcription pure."""
        lang_map = {
            "french": "français",
            "english": "English"
        }
        lang_name = lang_map.get(language, "français")
        
        return f"Transcris fidèlement cet audio en {lang_name}. Inclus les noms des locuteurs si identifiables."
    
    
    def _parse_comprehension_response(self, response: str) -> Tuple[str, str]:
        """Parse la réponse du mode compréhension."""
        # Simple parsing basé sur les sections
        sections = response.split("2. RÉSUMÉ" if "RÉSUMÉ" in response else "2. SUMMARY")
        
        if len(sections) >= 2:
            transcription = sections[0].replace("1. TRANSCRIPTION", "").strip()
            summary_part = sections[1].split("3. ACTIONS")[0].strip()
            actions_part = ""
            
            if "3. ACTIONS" in response:
                actions_part = response.split("3. ACTIONS")[1].strip()
            
            summary = summary_part
            if actions_part:
                summary += f"\n\n**Actions identifiées :**\n{actions_part}"
                
            return transcription, summary
        else:
            # Fallback si le parsing échoue
            return response, "Résumé non disponible"
    
    def _merge_summaries(self, summaries: List[str]) -> str:
        """Fusionne les résumés de plusieurs chunks avec gestion mémoire optimisée."""
        if not summaries:
            return ""
        
        if len(summaries) == 1:
            return summaries[0]
        
        # Utiliser un générateur pour éviter la concatenation multiple en mémoire
        def summary_generator():
            yield "# Résumé de la réunion\n\n"
            for i, summary in enumerate(summaries):
                yield f"## Partie {i+1}\n{summary}\n\n"
        
        # Jointure efficace avec un buffer
        try:
            merged = "".join(summary_generator())
            return merged.strip()
        finally:
            # Nettoyage des références
            summaries.clear() if hasattr(summaries, 'clear') else None
    

    def analyze_audio_chunks(
        self, 
        wav_path: str, 
        language: str = "french", 
        selected_sections: list = None,
        chunk_duration_minutes: int = 15,
        reference_speakers_data=None,
        progress_callback=None
    ) -> Dict[str, str]:
        """
        Analyse directe de l'audio par chunks via audio instruct mode.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            language (str): Langue attendue
            selected_sections (list): Sections d'analyse à inclure
            chunk_duration_minutes (int): Durée des chunks en minutes
            reference_speakers_data: Données de diarisation
            progress_callback: Fonction de callback pour le suivi de progression
            
        Returns:
            Dict[str, str]: Résultats avec 'transcription' (analyse concaténée)
        """
        # Mesurer le temps total de traitement
        total_start_time = time.time()
        
        duration = self._get_audio_duration(wav_path)
        print(f"🎵 Durée audio: {duration:.1f} minutes")
        
        # Créer des chunks de temps fixes avec overlap de 10 secondes
        chunk_duration = chunk_duration_minutes * 60  # Convertir en secondes
        overlap_duration = 10.0  # 10 secondes d'overlap
        chunks = []
        current_time = 0
        
        while current_time < duration * 60:
            end_time = min(current_time + chunk_duration, duration * 60)
            chunks.append((current_time, end_time))
            
            # Pour le prochain chunk, commencer 10 secondes avant la fin du chunk actuel
            # sauf si c'est le dernier chunk
            if end_time < duration * 60:
                current_time = max(0, end_time - overlap_duration)
            else:
                break
        
        print(f"📦 Division en {len(chunks)} chunks de {chunk_duration_minutes} minutes")
        
        # Calculer le nombre total d'étapes pour la progression (chunks + synthèse si plusieurs chunks)
        total_steps = len(chunks) + (1 if len(chunks) > 1 else 0)
        
        # Liste pour stocker les résumés de chaque chunk
        chunk_summaries = []
        
        for i, (start_time, end_time) in enumerate(chunks):
            print(f"🎯 Traitement du chunk {i+1}/{len(chunks)} ({start_time/60:.1f}-{end_time/60:.1f}min)")
            
            # Mise à jour de la progression
            if progress_callback:
                progress_callback((i / total_steps), f"Analyse du chunk {i+1}/{len(chunks)}")
            
            MemoryManager.print_memory_stats(f"Avant chunk {i+1}")
            
            # Mesurer le temps de traitement du chunk
            chunk_start_time = time.time()
            
            # Extraire le chunk audio
            chunk_path = self._extract_audio_chunk(wav_path, start_time, end_time)
            
            try:
                # Ajuster la diarisation selon l'offset temporel du chunk
                adjusted_speaker_context = ""
                if reference_speakers_data:
                    chunk_duration_sec = end_time - start_time
                    adjusted_speaker_context = self._adjust_diarization_timestamps(reference_speakers_data, start_time, chunk_duration_sec)
                
                # Utiliser audio instruct mode pour analyse directe depuis la config centralisée
                sections_list = selected_sections if selected_sections else ["resume_executif"]
                
                # Indiquer que c'est un segment d'un audio plus long seulement s'il y a plusieurs chunks
                chunk_info = f"SEGMENT {i+1}/{len(chunks)} ({start_time/60:.1f}-{end_time/60:.1f}min)" if len(chunks) > 1 else None
                prompt_text = VoxtralPrompts.get_meeting_summary_prompt(sections_list, adjusted_speaker_context, chunk_info, None)
                
                

                conversation = [{
                    "role": "user", 
                    "content": [
                        {"type": "audio", "path": chunk_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }]
                
                # Utiliser apply_chat_template pour l'audio instruct mode
                inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
                inputs = inputs.to(self.device, dtype=self.dtype)
                
                # Génération avec tokens pour résumés
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=16000,  # Augmenté pour résumés détaillés comme l'API
                        temperature=0.2,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        output_scores=False
                    )
                
                # Décodage du résumé
                input_tokens = inputs.input_ids.shape[1]
                output_tokens_count = outputs.shape[1] - input_tokens
                
                chunk_summary = self.processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
                
                token_tracker.add_chunk_tokens(input_tokens, output_tokens_count)
                
                chunk_summaries.append(f"## Segment {i+1} ({start_time/60:.1f}-{end_time/60:.1f}min)\n\n{chunk_summary}")
                
                # Plus de contexte cumulé
                
                # Calculer et afficher le temps de traitement
                chunk_duration = time.time() - chunk_start_time
                print(f"✅ Chunk {i+1} analysé en {format_duration(chunk_duration)}: {len(chunk_summary)} caractères")
                
            except Exception as e:
                print(f"❌ Erreur chunk {i+1}: {e}")
                chunk_summaries.append(f"**Segment {i+1}:** Erreur de traitement")
            finally:
                # Nettoyer le fichier temporaire du chunk
                import os
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        # Traitement final selon le nombre de chunks
        if chunk_summaries:
            if len(chunk_summaries) == 1:
                # Un seul chunk, pas besoin de synthèse
                final_analysis = chunk_summaries[0]
                
                # Progression complète pour un chunk unique
                if progress_callback:
                    progress_callback(1.0, "Analyse terminée !")
                    
                print(f"✅ Analyse directe terminée")
            else:
                # Plusieurs chunks : synthèse finale en mode texte
                print(f"🔄 Synthèse finale en mode texte des {len(chunk_summaries)} segments...")
                
                # Mise à jour de la progression pour la synthèse
                if progress_callback:
                    progress_callback((len(chunks) / total_steps), "Synthèse finale en cours...")
                
                combined_content = "\n\n".join(chunk_summaries)
                final_analysis = self._synthesize_chunks_final(combined_content, selected_sections)
                
                # Progression complète après synthèse
                if progress_callback:
                    progress_callback(1.0, "Analyse terminée !")
                    
                print(f"✅ Analyse avec synthèse finale terminée avec {len(chunk_summaries)} segments")
        else:
            final_analysis = "Aucune analyse disponible."
        
        # Calculer et afficher le temps total
        total_duration = time.time() - total_start_time
        print(f"⏱️ Analyse directe totale en {format_duration(total_duration)} pour {duration:.1f}min d'audio")
        
        # Print token usage summary
        token_tracker.print_summary()
        
        # Nettoyage final des fichiers temporaires
        cleanup_temp_files()
        
        return {"transcription": final_analysis}
    
    def _synthesize_chunks_final(self, combined_content: str, selected_sections: list) -> str:
        """
        Fait une synthèse finale en mode texte de tous les segments analysés.
        
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
                        print(f"✅ Section ajoutée à la synthèse: {section['title']}")
            
            synthesis_prompt = f"""Voici les analyses détaillées de plusieurs segments d'une réunion :

{combined_content}

INSTRUCTION CRITIQUE - LANGUE DE RÉPONSE :
- DÉTECTE la langue utilisée dans les segments ci-dessus
- RÉPONDS OBLIGATOIREMENT dans cette même langue détectée
- Si les segments sont en français → réponds en français
- Si les segments sont en anglais → réponds en anglais

Synthétise maintenant ces analyses en un résumé global cohérent et structuré selon les sections demandées :{sections_text}

Fournis une synthèse unifiée qui combine et résume les informations de tous les segments de manière cohérente."""

            
            # Vérifier que le modèle est encore disponible
            if not hasattr(self, 'model') or self.model is None:
                raise Exception("Le modèle n'est plus disponible (libéré de la mémoire)")
                
            print(f"✅ Modèle disponible: {type(self.model)}")
            print(f"✅ Processor disponible: {type(self.processor)}")

            # Générer la synthèse avec le modèle en mode texte
            conversation = [{"role": "user", "content": synthesis_prompt}]
            
            print(f"🔄 Application du chat template...")
            inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
            print(f"✅ Inputs type: {type(inputs)}")
            print(f"✅ Inputs keys: {inputs.keys() if hasattr(inputs, 'keys') else 'no keys'}")
            
            # Déplacer sur le device approprié
            if hasattr(inputs, 'keys'):  # BatchFeature ou dict
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                if 'input_ids' in inputs:
                    print(f"✅ Input_ids shape: {inputs['input_ids'].shape}")
            else:
                inputs = inputs.to(self.device)
                print(f"✅ Inputs shape: {inputs.shape}")
            
            print(f"🔄 Génération en cours...")
            outputs = self.model.generate(
                **inputs if hasattr(inputs, 'keys') else inputs,
                max_new_tokens=4000,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            print(f"✅ Outputs shape: {outputs.shape}")
            
            print(f"🔄 Décodage de la réponse...")
            # Déterminer la longueur des inputs pour le décodage
            if hasattr(inputs, 'keys') and 'input_ids' in inputs:
                input_length = inputs['input_ids'].shape[1]
            else:
                input_length = inputs.shape[1]
            
            output_tokens_count = outputs.shape[1] - input_length
            final_synthesis = self.processor.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            token_tracker.add_synthesis_tokens(input_length, output_tokens_count)
            
            return f"# Résumé Global de Réunion\n\n{final_synthesis}\n\n---\n\n## Détails par Segment\n\n{combined_content}"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Erreur détaillée lors de la synthèse finale:")
            print(error_details)
            return f"# Résumé de Réunion\n\n⚠️ Erreur lors de la synthèse finale: {str(e)}\n\n## Analyses par Segment\n\n{combined_content}"
    
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
        Analyse les locuteurs d'un fichier audio et génère un tableau des prises de parole.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            num_speakers (int, optional): Nombre de locuteurs attendu
            
        Returns:
            str: Tableau markdown des prises de parole par locuteur
        """
        print(f"🎤 Analyse des locuteurs avec Voxtral local...")
        
        # Limiter l'audio aux 10 premières minutes
        limited_wav_path = self._limit_audio_to_ten_minutes(wav_path)
        
        # Mesurer le temps de traitement
        start_time = time.time()
        
        try:
            # Cette fonctionnalité a été supprimée
            return "⚠️ Fonctionnalité d'identification des locuteurs supprimée"

            # Utiliser l'audio instruct mode avec le prompt personnalisé
            conversation = [{
                "role": "user", 
                "content": [
                    {"type": "audio", "path": limited_wav_path},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            
            # Utiliser apply_chat_template pour l'audio instruct mode
            inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
            inputs = inputs.to(self.device, dtype=self.dtype)
            
            # Génération avec tokens pour analyse des locuteurs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=32000,  # 32k tokens pour analyses détaillées
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    output_scores=False
                )
            
            # Décodage du résultat
            analysis_result = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0].strip()
            
            # Parser le tableau et créer les extraits audio
            if analysis_result and not analysis_result.startswith("❌"):
                enriched_result = self._parse_speakers_table_and_create_snippets(analysis_result, limited_wav_path)
            else:
                enriched_result = analysis_result
            
            # Calculer et afficher le temps de traitement
            duration = time.time() - start_time
            print(f"✅ Analyse des locuteurs terminée en {format_duration(duration)}")
            
            return enriched_result
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse des locuteurs: {e}")
            return f"❌ Erreur lors de l'analyse des locuteurs: {str(e)}"
        
        finally:
            # Nettoyer le fichier temporaire limité s'il a été créé
            if limited_wav_path != wav_path and os.path.exists(limited_wav_path):
                try:
                    os.unlink(limited_wav_path)
                    print(f"🧹 Fichier temporaire limité supprimé: {limited_wav_path}")
                except:
                    pass
            
            # Nettoyage de la mémoire
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            MemoryManager.full_cleanup()

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
        Libère complètement le modèle de la mémoire.
        À utiliser après traitement pour libérer la mémoire GPU/MPS.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Déplacer le modèle vers CPU avant suppression
            self.model.to('cpu')
            del self.model
            self.model = None
            
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None
            
        # Nettoyage forcé
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("🧹 Modèle Voxtral libéré de la mémoire")
        MemoryManager.print_memory_stats("Après libération modèle Voxtral")