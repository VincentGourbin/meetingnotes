"""
Speaker diarization module for HF Spaces with Zero GPU support.

This module uses pyannote/speaker-diarization-3.1 to identify 
and segment different speakers in an audio file, optimized for HF Spaces.
"""

import torch
import torchaudio
from pyannote.audio import Pipeline
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import os
from pydub import AudioSegment
import time

from ..utils.zero_gpu_manager import gpu_model_loading, gpu_inference, ZeroGPUManager


class SpeakerDiarization:
    """
    Speaker diarization using pyannote/speaker-diarization-3.1 for HF Spaces.
    
    This class handles automatic speaker diarization
    with Zero GPU decorators for efficient compute allocation.
    """
    
    def __init__(self, hf_token: str = None):
        """
        Initialize the pyannote diarizer for HF Spaces.
        
        Args:
            hf_token (str): Hugging Face token to access the model
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.pipeline = None
        self.gpu_manager = ZeroGPUManager()
        print("🔄 Initializing pyannote diarizer for HF Spaces...")
        
    @gpu_model_loading(duration=90)
    def _load_pipeline(self):
        """Load diarization pipeline with GPU allocation if not already loaded."""
        if self.pipeline is None:
            print("📥 Loading pyannote/speaker-diarization-3.1 model...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            # Use GPU if available (CUDA or MPS)
            if self.gpu_manager.is_gpu_available():
                device = self.gpu_manager.get_device()
                if device == "mps":
                    # MPS support for local Mac testing
                    self.pipeline = self.pipeline.to(torch.device("mps"))
                    print("🚀 Pyannote pipeline loaded on MPS (Apple Silicon)")
                elif device == "cuda":
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                    print("🚀 Pyannote pipeline loaded on CUDA")
                else:
                    print("⚠️ Pyannote pipeline loaded on CPU")
            else:
                print("⚠️ Pyannote pipeline loaded on CPU")
    
    @gpu_inference(duration=180)
    def diarize_audio(self, audio_path: str, num_speakers: Optional[int] = None) -> Tuple[str, List[Dict]]:
        """
        Perform speaker diarization on an audio file with Zero GPU.
        
        Args:
            audio_path (str): Path to the audio file
            num_speakers (Optional[int]): Expected number of speakers (optional)
            
        Returns:
            Tuple[str, List[Dict]]: (RTTM result, List of reference segments for each speaker)
        """
        try:
            # Load pipeline if necessary
            self._load_pipeline()
            
            print(f"🎤 Starting diarization: {audio_path}")
            
            # Prepare audio file for pyannote (mono WAV)
            processed_audio_path = self._prepare_audio_for_pyannote(audio_path)
            
            # Diarization parameters
            diarization_params = {}
            if num_speakers is not None:
                diarization_params["num_speakers"] = num_speakers
                print(f"👥 Specified number of speakers: {num_speakers}")
            
            # Perform diarization
            print("🔍 Speaker analysis in progress...")
            diarization = self.pipeline(processed_audio_path, **diarization_params)
            
            # Convert to RTTM format
            rttm_output = self._convert_to_rttm(diarization, audio_path)
            
            # Extract reference segments (first long segments for each speaker)
            try:
                reference_segments = self._extract_reference_segments(diarization, audio_path, min_duration=5.0)
            except Exception as ref_error:
                print(f"⚠️ Error extracting reference segments: {ref_error}")
                reference_segments = []
            
            print(f"✅ Diarization completed: {len(diarization)} segments detected")
            print(f"🎤 Reference segments created: {len(reference_segments)} speakers")
            
            # Clean up temporary file if created
            if processed_audio_path != audio_path:
                try:
                    os.unlink(processed_audio_path)
                except:
                    pass
            
            return rttm_output, reference_segments
            
        except Exception as e:
            print(f"❌ Error during diarization: {e}")
            return f"❌ Error during diarization: {str(e)}", []
        finally:
            # Clean up GPU memory
            self.gpu_manager.cleanup_gpu()
    
    def _prepare_audio_for_pyannote(self, audio_path: str) -> str:
        """
        Prepare audio file for pyannote (mono WAV if necessary).
        
        Args:
            audio_path (str): Path to original audio file
            
        Returns:
            str: Path to prepared audio file
        """
        try:
            # Load audio with pydub to check format
            audio = AudioSegment.from_file(audio_path)
            
            # Check if conversion is needed (mono + WAV)
            needs_conversion = (
                audio.channels != 1 or  # Not mono
                not audio_path.lower().endswith('.wav')  # Not WAV
            )
            
            if not needs_conversion:
                print("🎵 Audio already in correct format for pyannote")
                return audio_path
            
            print("🔄 Converting audio for pyannote (mono WAV)...")
            
            # Convert to mono WAV
            mono_audio = audio.set_channels(1)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Export as mono WAV
            mono_audio.export(temp_path, format="wav")
            
            print(f"✅ Audio converted: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"⚠️ Audio conversion error: {e}, using original file")
            return audio_path
    
    def _convert_to_rttm(self, diarization, audio_file: str) -> str:
        """
        Convert diarization result to RTTM format.
        
        Args:
            diarization: Pyannote diarization object
            audio_file (str): Audio filename for RTTM
            
        Returns:
            str: RTTM format content
        """
        rttm_lines = []
        
        # RTTM header
        audio_filename = os.path.basename(audio_file)
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # RTTM format: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
            start_time = segment.start
            duration = segment.duration
            
            rttm_line = f"SPEAKER {audio_filename} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
            rttm_lines.append(rttm_line)
        
        return "\n".join(rttm_lines)
    
    def _extract_reference_segments(self, diarization, audio_path: str, min_duration: float = 5.0) -> List[Dict]:
        """
        Extract first long segment for each speaker as reference.
        
        Args:
            diarization: Pyannote diarization object
            audio_path (str): Path to audio file
            min_duration (float): Minimum duration in seconds for a reference segment
            
        Returns:
            List[Dict]: List of reference segments with metadata
        """
        reference_segments = []
        speakers_found = set()
        
        print(f"🔍 Searching for reference segments (>{min_duration}s) for each speaker...")
        
        # Iterate through all segments to find first long segment of each speaker
        try:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers_found and segment.duration >= min_duration:
                    print(f"👤 {speaker}: {segment.duration:.1f}s segment found ({segment.start:.1f}s-{segment.end:.1f}s)")
                    
                    # Create audio snippet
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
            
            # Fallback: if no long segments found for some speakers, take the longest
            all_speakers_in_diarization = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
            if len(speakers_found) < len(all_speakers_in_diarization):
                print("⚠️ Some speakers don't have long segments, using longest segments...")
                self._add_fallback_segments(diarization, audio_path, reference_segments, speakers_found, min_duration)
                
        except Exception as iter_error:
            print(f"❌ Error iterating segments: {iter_error}")
            reference_segments = []
        
        return reference_segments
    
    def _add_fallback_segments(self, diarization, audio_path: str, reference_segments: List[Dict], 
                              speakers_found: set, min_duration: float):
        """Add fallback segments for speakers without long segments."""
        all_speakers = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
        missing_speakers = all_speakers - speakers_found
        
        for speaker in missing_speakers:
            # Find longest segment for this speaker
            longest_segment = None
            longest_duration = 0
            
            for segment, _, spk in diarization.itertracks(yield_label=True):
                if spk == speaker and segment.duration > longest_duration:
                    longest_segment = segment
                    longest_duration = segment.duration
            
            if longest_segment and longest_duration > 1.0:  # At least 1 second
                print(f"👤 {speaker}: fallback segment of {longest_duration:.1f}s")
                
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
        Create temporary audio snippet for a speaker segment.
        
        Args:
            audio_path (str): Path to source audio file
            start_time (float): Start in seconds
            end_time (float): End in seconds
            speaker (str): Speaker ID
            
        Returns:
            Optional[str]: Path to created temporary audio snippet or None if error
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Extract segment
            segment = audio[start_ms:end_ms]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=f"_{speaker}_{start_time:.1f}s.wav", 
                delete=False
            ) as tmp_file:
                snippet_path = tmp_file.name
            
            # Export snippet to temporary file
            segment.export(snippet_path, format="wav")
            
            print(f"🎵 Temporary snippet created: {snippet_path}")
            return snippet_path
            
        except Exception as e:
            print(f"❌ Error creating snippet for {speaker}: {e}")
            return None
    
    def cleanup(self):
        """Release pipeline resources."""
        if self.pipeline is not None:
            # Free GPU/MPS memory by moving to CPU
            if hasattr(self.pipeline, 'to'):
                try:
                    self.pipeline = self.pipeline.to(torch.device('cpu'))
                except Exception as e:
                    print(f"⚠️ Error moving to CPU: {e}")
            
            del self.pipeline
            self.pipeline = None
            
            # Clean up memory
            self.gpu_manager.cleanup_gpu()
            print("🧹 Pyannote pipeline freed from memory")