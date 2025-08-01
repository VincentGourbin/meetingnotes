"""
Module d'analyse audio avec les modèles Voxtral via MLX (optimisé Apple Silicon).

Ce module utilise MLX pour :
- L'analyse audio multilingue directe optimisée pour Apple Silicon
- La génération de résumés structurés depuis l'audio
- Le traitement des fichiers longs avec découpage automatique
- Performance accélérée sur Mac M1/M2/M3

Dépendances:
    - mlx: Framework ML optimisé Apple Silicon
    - mlx-voxtral: Implémentation MLX des modèles Voxtral
    - pydub: Traitement audio pour le découpage
"""

import os
import tempfile
import time
from pydub import AudioSegment
from typing import List, Dict, Tuple, Optional
import math

from .prompts_config import VoxtralPrompts
from ..utils import format_duration, token_tracker


class VoxtralMLXAnalyzer:
    """
    Analyseur audio intelligent utilisant les modèles Voxtral via MLX.
    
    Cette classe gère l'analyse et compréhension audio directe via MLX
    avec support du découpage intelligent pour les fichiers longs.
    Optimisé pour les processeurs Apple Silicon.
    """
    
    def __init__(self, model_name: str = "mistralai/Voxtral-Mini-3B-2507"):
        """
        Initialise l'analyseur MLX Voxtral.
        
        Args:
            model_name (str): Nom du modèle ("mistralai/Voxtral-Mini-3B-2507" ou "mistralai/Voxtral-Small-24B-2507")
        """
        self.model_name = model_name
        self.max_duration_minutes = 15  # Limite raisonnable pour éviter les timeouts
        
        # Mapping des noms de modèles
        model_mapping = {
            "Voxtral-Mini-3B-2507": "mistralai/Voxtral-Mini-3B-2507",
            "Voxtral-Small-24B-2507": "mistralai/Voxtral-Small-24B-2507"
        }
        
        self.model_name = model_mapping.get(model_name, model_name)
        
        self.model = None
        self.processor = None
        
        print(f"🔄 Initialisation de l'analyseur MLX Voxtral...")
        print(f"🤖 Modèle: {self.model_name}")
        print(f"🚀 Backend: MLX (Apple Silicon optimisé)")
    
    def _load_model(self):
        """
        Charge le modèle Voxtral avec MLX.
        """
        if self.model is None or self.processor is None:
            try:
                from mlx_voxtral import load_voxtral_model, VoxtralProcessor
                import mlx.core as mx
                
                print(f"🔄 Chargement du modèle MLX...")
                
                # Utiliser directement le nom du modèle (peut être quantifié)
                mlx_model_id = self.model_name
                print(f"🔄 Utilisation du modèle MLX: {mlx_model_id}")
                
                self.model, self.config = load_voxtral_model(mlx_model_id, dtype=mx.bfloat16)
                self.processor = VoxtralProcessor.from_pretrained(mlx_model_id)
                
                print(f"✅ Modèle MLX chargé avec succès")
                
            except ImportError as e:
                raise ImportError(f"Erreur d'import mlx-voxtral. Vérifiez l'installation:\npip install mlx mlx-voxtral\nErreur: {e}")
            except Exception as e:
                raise RuntimeError(f"Erreur lors du chargement du modèle MLX: {e}")
    
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
    
    def _analyze_audio_chunk_mlx(self, audio_path: str, prompt: str) -> str:
        """
        Analyse directe d'un chunk audio via MLX avec inférence chat multimodale.
        
        Utilise l'approche directe audio-chat comme dans l'exemple voxtral_chat.py
        pour éviter la transcription intermédiaire.
        
        Args:
            audio_path (str): Chemin vers le fichier audio
            prompt (str): Prompt d'analyse
            
        Returns:
            str: Résultat de l'analyse
        """
        try:
            # S'assurer que le modèle est chargé
            self._load_model()
            
            if not os.path.exists(audio_path):
                return f"❌ Fichier audio non trouvé: {audio_path}"
            
            # Construire la conversation avec contenu multimodal (texte + audio)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "url": audio_path}
                    ]
                }
            ]
            
            # Appliquer le template de chat pour formater les inputs
            inputs = self.processor.apply_chat_template(conversation, return_tensors="mlx")
            
            # Générer la réponse avec le modèle MLX
            outputs = self.model.generate(
                inputs["input_ids"],
                input_features=inputs.get("input_features"),  # Features audio si disponibles
                max_new_tokens=4000,
                do_sample=False,  # Déterministe pour plus de cohérence
                temperature=0.1
            )
            # Décoder la réponse en excluant les tokens d'entrée
            # Convertir de MLX vers numpy/list avant décodage
            import mlx.core as mx
            input_tokens_count = inputs["input_ids"].shape[1]
            output_tokens = outputs[0][input_tokens_count:]
            output_tokens_count = len(output_tokens)
            output_tokens_numpy = mx.array(output_tokens).tolist()
            
            response = self.processor.tokenizer.decode(
                output_tokens_numpy,
                skip_special_tokens=True
            ).strip()
            
            token_tracker.add_chunk_tokens(input_tokens_count, output_tokens_count)
            
            if not response or len(response.strip()) < 10:
                return "❌ Réponse vide ou trop courte du modèle MLX"
            
            return response
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Erreur détaillée MLX:")
            print(error_details)
            return f"❌ Erreur analyse MLX: {str(e)}"
    
    def analyze_audio_chunks_mlx(
        self, 
        wav_path: str, 
        language: str = "french", 
        selected_sections: list = None,
        chunk_duration_minutes: int = 15,
        reference_speakers_data=None
    ) -> Dict[str, str]:
        """
        Analyse directe de l'audio par chunks via les modèles MLX.
        
        Args:
            wav_path (str): Chemin vers le fichier audio
            language (str): Langue attendue
            selected_sections (list): Sections du résumé à inclure
            chunk_duration_minutes (int): Durée des chunks en minutes
            reference_speakers_data: Données des locuteurs de référence identifiés
            
        Returns:
            Dict[str, str]: Résultats avec 'transcription' (analyse concaténée)
        """
        # Mesurer le temps total de traitement
        total_start_time = time.time()
        
        # Initialize token tracker for MLX mode
        token_tracker.reset()
        token_tracker.set_mode("MLX")
        
        # Obtenir la durée audio
        duration = self._get_audio_duration(wav_path)
        print(f"🎵 Durée audio: {duration:.1f} minutes")
        
        # Si le fichier est plus court que la limite, traiter en un seul chunk
        if duration <= chunk_duration_minutes:
            print(f"📄 Fichier court, traitement en un seul chunk")
            
            # Ajuster les timestamps de diarisation pour le fichier complet
            adjusted_speaker_context = ""
            if reference_speakers_data:
                adjusted_speaker_context = self._adjust_diarization_timestamps(
                    reference_speakers_data, 0, duration * 60
                )
            else:
                adjusted_speaker_context = ""
            
            # Utiliser audio instruct mode pour analyse directe depuis la config centralisée
            sections_list = selected_sections if selected_sections else ["resume_executif"]
            
            # Pas de chunk_info pour un fichier complet
            prompt_text = VoxtralPrompts.get_meeting_summary_prompt(sections_list, adjusted_speaker_context, None, None)
            
            print(f"🔄 Début analyse MLX du fichier {wav_path}")
            chunk_start_time = time.time()
            chunk_summary = self._analyze_audio_chunk_mlx(wav_path, prompt_text)
            chunk_duration = time.time() - chunk_start_time
            
            print(f"✅ Analyse terminée en {format_duration(chunk_duration)}")
            
            # Calculer et afficher le temps total
            total_duration = time.time() - total_start_time
            print(f"⏱️ Analyse directe MLX totale en {format_duration(total_duration)} pour {duration:.1f}min d'audio")
            
            return {"transcription": chunk_summary}
        
        # Créer des chunks de temps fixes avec overlap de 10 secondes
        chunk_duration = chunk_duration_minutes * 60  # Convertir en secondes
        overlap_duration = 10  # 10 secondes d'overlap
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
        
        # Liste pour stocker les résumés de chaque chunk
        chunk_summaries = []
        
        for i, (start_time, end_time) in enumerate(chunks):
            print(f"🎯 Traitement du chunk {i+1}/{len(chunks)} ({start_time/60:.1f}-{end_time/60:.1f}min)")
            
            # Mesurer le temps de traitement du chunk
            chunk_start_time = time.time()
            
            # Extraire le chunk audio
            chunk_path = self._extract_audio_chunk(wav_path, start_time, end_time)
            
            try:
                # Ajuster la diarisation selon l'offset temporel du chunk
                adjusted_speaker_context = ""
                if reference_speakers_data:
                    chunk_duration_seconds = end_time - start_time
                    adjusted_speaker_context = self._adjust_diarization_timestamps(
                        reference_speakers_data, start_time, chunk_duration_seconds
                    )
                
                # Utiliser audio instruct mode pour analyse directe depuis la config centralisée
                sections_list = selected_sections if selected_sections else ["resume_executif"]
                
                # Indiquer que c'est un segment d'un audio plus long seulement s'il y a plusieurs chunks
                chunk_info = f"SEGMENT {i+1}/{len(chunks)} ({start_time/60:.1f}-{end_time/60:.1f}min)" if len(chunks) > 1 else None
                prompt_text = VoxtralPrompts.get_meeting_summary_prompt(sections_list, adjusted_speaker_context, chunk_info, None)
                
                
                chunk_summary = self._analyze_audio_chunk_mlx(chunk_path, prompt_text)
                
                chunk_summaries.append(f"## Segment {i+1} ({start_time/60:.1f}-{end_time/60:.1f}min)\n\n{chunk_summary}")
                
                # Calculer et afficher le temps de traitement
                chunk_duration = time.time() - chunk_start_time
                print(f"✅ Chunk {i+1} analysé en {format_duration(chunk_duration)}")
                
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
                print(f"✅ Analyse directe MLX terminée")
            else:
                # Plusieurs chunks : synthèse finale en mode texte
                print(f"🔄 Synthèse finale MLX en mode texte des {len(chunk_summaries)} segments...")
                combined_content = "\n\n".join(chunk_summaries)
                final_analysis = self._synthesize_chunks_final_mlx(combined_content, selected_sections)
                print(f"✅ Analyse MLX avec synthèse finale terminée avec {len(chunk_summaries)} segments")
        else:
            final_analysis = "Aucune analyse disponible."
        
        # Calculer et afficher le temps total
        total_duration = time.time() - total_start_time
        print(f"⏱️ Analyse directe MLX totale en {format_duration(total_duration)} pour {duration:.1f}min d'audio")
        
        # Print token usage summary
        token_tracker.print_summary()
        
        return {"transcription": final_analysis}
    
    def _synthesize_chunks_final_mlx(self, combined_content: str, selected_sections: list) -> str:
        """
        Fait une synthèse finale en mode texte via MLX de tous les segments analysés.
        
        Args:
            combined_content (str): Contenu combiné de tous les segments
            selected_sections (list): Sections sélectionnées pour le résumé
            
        Returns:
            str: Synthèse finale structurée
        """
        try:
            # S'assurer que le modèle est chargé
            self._load_model()
            
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

            # Construire la conversation pour la synthèse textuelle 
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": synthesis_prompt}
                    ]
                }
            ]
            
            # Appliquer le template de chat
            inputs = self.processor.apply_chat_template(conversation, return_tensors="mlx")
            
            # Générer la synthèse avec MLX
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=4000,
                do_sample=False,
                temperature=0.1
            )
            
            # Décoder la réponse
            # Convertir de MLX vers numpy/list avant décodage
            import mlx.core as mx
            input_tokens_count = inputs["input_ids"].shape[1]
            synthesis_tokens = outputs[0][input_tokens_count:]
            synthesis_tokens_count = len(synthesis_tokens)
            synthesis_tokens_numpy = mx.array(synthesis_tokens).tolist()
            
            final_synthesis = self.processor.tokenizer.decode(
                synthesis_tokens_numpy,
                skip_special_tokens=True
            ).strip()
            
            token_tracker.add_synthesis_tokens(input_tokens_count, synthesis_tokens_count)
            
            return f"# Résumé Global de Réunion\n\n{final_synthesis}\n\n---\n\n## Détails par Segment\n\n{combined_content}"
            
        except Exception as e:
            print(f"❌ Erreur lors de la synthèse finale MLX: {e}")
            return f"# Résumé de Réunion\n\n⚠️ Erreur lors de la synthèse finale MLX: {str(e)}\n\n## Analyses par Segment\n\n{combined_content}"
    
    def cleanup_model(self):
        """
        Nettoie les ressources du modèle.
        """
        if self.model is not None:
            print("🧹 Nettoyage du modèle MLX...")
            # MLX gère automatiquement la mémoire
            self.model = None
            self.processor = None
            print("✅ Modèle MLX nettoyé")