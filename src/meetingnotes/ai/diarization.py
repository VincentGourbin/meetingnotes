"""
Module de diarisation audio avec pyannote.

Ce module utilise pyannote/speaker-diarization-3.1 pour identifier 
et segmenter les différents locuteurs dans un fichier audio.
"""

import torch
import torchaudio
from pyannote.audio import Pipeline
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import os
from pydub import AudioSegment
import time


class PyAnnoteDiarizer:
    """
    Diariseur utilisant pyannote/speaker-diarization-3.1.
    
    Cette classe gère la diarisation automatique des locuteurs
    avec le modèle pyannote pre-entraîné.
    """
    
    def __init__(self, hf_token: str):
        """
        Initialise le diariseur pyannote.
        
        Args:
            hf_token (str): Token Hugging Face pour accéder au modèle
        """
        self.hf_token = hf_token
        self.pipeline = None
        self.device = None
        print("🔄 Initialisation du diariseur pyannote...")
        
    def _load_pipeline(self):
        """Charge le pipeline de diarisation si pas encore fait."""
        if self.pipeline is None:
            print("📥 Chargement du modèle pyannote/speaker-diarization-3.1...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            # Utiliser MPS en priorité (Apple Silicon), puis CUDA, puis CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.pipeline = self.pipeline.to(self.device)
                print("🚀 Pipeline pyannote chargé sur MPS (Apple Silicon)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.pipeline = self.pipeline.to(self.device)
                print("🚀 Pipeline pyannote chargé sur GPU CUDA")
            else:
                self.device = torch.device("cpu")
                print("⚠️ Pipeline pyannote chargé sur CPU")
    
    def diarize_audio(self, audio_path: str, num_speakers: Optional[int] = None) -> Tuple[str, List[Dict]]:
        """
        Effectue la diarisation d'un fichier audio.
        
        Args:
            audio_path (str): Chemin vers le fichier audio
            num_speakers (Optional[int]): Nombre de locuteurs attendu (optionnel)
            
        Returns:
            Tuple[str, List[Dict]]: (Résultat RTTM, Liste des segments de référence pour chaque locuteur)
        """
        try:
            # Charger le pipeline si nécessaire
            self._load_pipeline()
            
            print(f"🎤 Démarrage de la diarisation: {audio_path}")
            
            # Préparer le fichier audio pour pyannote (WAV mono)
            processed_audio_path = self._prepare_audio_for_pyannote(audio_path)
            
            # Paramètres de diarisation
            diarization_params = {}
            if num_speakers is not None:
                diarization_params["num_speakers"] = num_speakers
                print(f"👥 Nombre de locuteurs spécifié: {num_speakers}")
            
            # Effectuer la diarisation
            print("🔍 Analyse des locuteurs en cours...")
            diarization = self.pipeline(processed_audio_path, **diarization_params)
            
            # Convertir en format RTTM
            rttm_output = self._convert_to_rttm(diarization, audio_path)
            
            # Extraire les segments de référence (premiers segments longs pour chaque locuteur)
            try:
                reference_segments = self._extract_reference_segments(diarization, audio_path, min_duration=5.0)
            except Exception as ref_error:
                print(f"⚠️ Erreur lors de l'extraction des segments de référence: {ref_error}")
                reference_segments = []
            
            print(f"✅ Diarisation terminée: {len(diarization)} segments détectés")
            print(f"🎤 Segments de référence créés: {len(reference_segments)} locuteurs")
            
            # Nettoyer le fichier temporaire si créé
            if processed_audio_path != audio_path:
                try:
                    os.unlink(processed_audio_path)
                except:
                    pass
            
            return rttm_output, reference_segments
            
        except Exception as e:
            print(f"❌ Erreur lors de la diarisation: {e}")
            return f"❌ Erreur lors de la diarisation: {str(e)}", []
    
    def _prepare_audio_for_pyannote(self, audio_path: str) -> str:
        """
        Prépare le fichier audio pour pyannote (WAV mono si nécessaire).
        
        Args:
            audio_path (str): Chemin vers le fichier audio original
            
        Returns:
            str: Chemin vers le fichier audio préparé
        """
        try:
            # Charger l'audio avec pydub pour vérifier le format
            audio = AudioSegment.from_file(audio_path)
            
            # Vérifier si conversion nécessaire (mono + WAV)
            needs_conversion = (
                audio.channels != 1 or  # Pas mono
                not audio_path.lower().endswith('.wav')  # Pas WAV
            )
            
            if not needs_conversion:
                print("🎵 Audio déjà au bon format pour pyannote")
                return audio_path
            
            print("🔄 Conversion audio pour pyannote (mono WAV)...")
            
            # Convertir en mono WAV
            mono_audio = audio.set_channels(1)
            
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Exporter en WAV mono
            mono_audio.export(temp_path, format="wav")
            
            print(f"✅ Audio converti: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"⚠️ Erreur conversion audio: {e}, utilisation fichier original")
            return audio_path
    
    def _convert_to_rttm(self, diarization, audio_file: str) -> str:
        """
        Convertit le résultat de diarisation au format RTTM.
        
        Args:
            diarization: Objet diarization de pyannote
            audio_file (str): Nom du fichier audio pour le RTTM
            
        Returns:
            str: Contenu au format RTTM
        """
        rttm_lines = []
        
        # En-tête RTTM
        audio_filename = os.path.basename(audio_file)
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # Format RTTM: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
            start_time = segment.start
            duration = segment.duration
            
            rttm_line = f"SPEAKER {audio_filename} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
            rttm_lines.append(rttm_line)
        
        return "\n".join(rttm_lines)
    
    def _extract_reference_segments(self, diarization, audio_path: str, min_duration: float = 5.0) -> List[Dict]:
        """
        Extrait le premier segment long pour chaque locuteur comme référence.
        
        Args:
            diarization: Objet diarization de pyannote
            audio_path (str): Chemin vers le fichier audio
            min_duration (float): Durée minimum en secondes pour un segment de référence
            
        Returns:
            List[Dict]: Liste des segments de référence avec métadonnées
        """
        reference_segments = []
        speakers_found = set()
        
        print(f"🔍 Recherche de segments de référence (>{min_duration}s) pour chaque locuteur...")
        
        # Parcourir tous les segments pour trouver le premier segment long de chaque locuteur
        try:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers_found and segment.duration >= min_duration:
                    print(f"👤 {speaker}: segment de {segment.duration:.1f}s trouvé ({segment.start:.1f}s-{segment.end:.1f}s)")
                    
                    # Créer l'extrait audio
                    snippet_path = self._create_audio_snippet(
                        audio_path, 
                        segment.start, 
                        segment.end, 
                        speaker
                    )
                    
                    if snippet_path:
                        reference_segments.append({
                            'speaker': speaker,
                            'start': segment.start,
                            'end': segment.end,
                            'duration': segment.duration,
                            'audio_path': snippet_path
                        })
                        speakers_found.add(speaker)
            
            # Fallback: si aucun segment long trouvé pour certains locuteurs, prendre le plus long
            all_speakers_in_diarization = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
            if len(speakers_found) < len(all_speakers_in_diarization):
                print("⚠️ Certains locuteurs n'ont pas de segments longs, utilisation des segments les plus longs...")
                self._add_fallback_segments(diarization, audio_path, reference_segments, speakers_found, min_duration)
                
        except Exception as iter_error:
            print(f"❌ Erreur lors du parcours des segments: {iter_error}")
            # En cas d'erreur, retourner au moins une liste vide
            reference_segments = []
        
        return reference_segments
    
    def _add_fallback_segments(self, diarization, audio_path: str, reference_segments: List[Dict], 
                              speakers_found: set, min_duration: float):
        """Ajoute des segments de fallback pour les locuteurs sans segments longs."""
        all_speakers = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
        missing_speakers = all_speakers - speakers_found
        
        for speaker in missing_speakers:
            # Trouver le segment le plus long pour ce locuteur
            longest_segment = None
            longest_duration = 0
            
            for segment, _, spk in diarization.itertracks(yield_label=True):
                if spk == speaker and segment.duration > longest_duration:
                    longest_segment = segment
                    longest_duration = segment.duration
            
            if longest_segment and longest_duration > 1.0:  # Au moins 1 seconde
                print(f"👤 {speaker}: segment de fallback de {longest_duration:.1f}s")
                
                snippet_path = self._create_audio_snippet(
                    audio_path,
                    longest_segment.start,
                    longest_segment.end,
                    speaker
                )
                
                if snippet_path:
                    reference_segments.append({
                        'speaker': speaker,
                        'start': longest_segment.start,
                        'end': longest_segment.end,
                        'duration': longest_duration,
                        'audio_path': snippet_path
                    })
    
    def _create_audio_snippet(self, audio_path: str, start_time: float, end_time: float, speaker: str) -> Optional[str]:
        """
        Crée un extrait audio temporaire pour un segment de locuteur.
        
        Args:
            audio_path (str): Chemin vers le fichier audio source
            start_time (float): Début en secondes
            end_time (float): Fin en secondes
            speaker (str): ID du locuteur
            
        Returns:
            Optional[str]: Chemin vers l'extrait audio temporaire créé ou None si erreur
        """
        try:
            # Charger l'audio
            audio = AudioSegment.from_file(audio_path)
            
            # Convertir en millisecondes
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Extraire le segment
            segment = audio[start_ms:end_ms]
            
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(
                suffix=f"_{speaker}_{start_time:.1f}s.wav", 
                delete=False
            ) as tmp_file:
                snippet_path = tmp_file.name
            
            # Exporter l'extrait vers le fichier temporaire
            segment.export(snippet_path, format="wav")
            
            print(f"🎵 Extrait temporaire créé: {snippet_path}")
            return snippet_path
            
        except Exception as e:
            print(f"❌ Erreur création extrait pour {speaker}: {e}")
            return None
    
    def cleanup(self):
        """
        Libère les ressources du pipeline.
        """
        if self.pipeline is not None:
            # Libérer la mémoire GPU/MPS en déplaçant vers CPU
            if hasattr(self.pipeline, 'to') and self.device is not None:
                try:
                    self.pipeline = self.pipeline.to(torch.device('cpu'))
                except Exception as e:
                    print(f"⚠️ Erreur lors du déplacement vers CPU: {e}")
            
            del self.pipeline
            self.pipeline = None
            self.device = None
            
            # Nettoyer la mémoire selon le device utilisé
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("🧹 Pipeline pyannote libéré de la mémoire")