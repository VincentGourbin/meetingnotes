"""
Voxtral analyzer optimized for Hugging Face Spaces.

This module provides audio analysis using Voxtral models with:
- Only Transformers backend (no MLX or API)
- Pre-quantized models for memory efficiency
- Zero GPU decorators for HF Spaces compute allocation
- Optimized memory management for Spaces environment
"""

import torch
import torchaudio
import tempfile
import time
import gc
import os
from transformers import VoxtralForConditionalGeneration, AutoProcessor, AutoConfig
from pydub import AudioSegment
from typing import List, Dict, Tuple, Optional

from ..utils.zero_gpu_manager import gpu_model_loading, gpu_inference, gpu_long_task, ZeroGPUManager
from .prompts_config import VoxtralPrompts
from ..utils.token_tracker import TokenTracker


class VoxtralSpacesAnalyzer:
    """
    Voxtral analyzer optimized for Hugging Face Spaces.
    
    Features:
    - Pre-quantized models for efficient memory usage
    - Zero GPU decorators for efficient compute allocation
    - Memory-optimized processing for Spaces constraints
    - On-demand model loading with caching
    """
    
    def __init__(self, model_name: str = "Voxtral-Mini-3B-2507"):
        """
        Initialize the Voxtral analyzer for HF Spaces.
        
        Args:
            model_name (str): Name of the Voxtral model to use (pre-quantized)
        """
        # Use original Mistral models for HF Spaces
        model_mapping = {
            "Voxtral-Mini-3B-2507": "mistralai/Voxtral-Mini-3B-2507",
            "Voxtral-Small-24B-2507": "mistralai/Voxtral-Small-24B-2507"
        }
        
        self.model_name = model_mapping.get(model_name, "mistralai/Voxtral-Mini-3B-2507")
        self.current_model_key = model_name
        
        # Optimized chunk durations for Zero GPU (different per model)
        self.chunk_durations = {
            "Voxtral-Mini-3B-2507": 15,  # 15 minutes for Mini
            "Voxtral-Small-24B-2507": 10   # 10 minutes for Small (larger model)
        }
        self.gpu_manager = ZeroGPUManager()
        self.token_tracker = TokenTracker("Transformers-HF-Spaces")
        
        # Model and processor will be loaded on-demand
        self.model = None
        self.processor = None
        
        print(f"🚀 VoxtralSpacesAnalyzer initialized for model: {model_name}")
    
    def switch_model(self, model_name: str):
        """Switch to a different model (will reload if different)."""
        model_mapping = {
            "Voxtral-Mini-3B-2507": "mistralai/Voxtral-Mini-3B-2507",
            "Voxtral-Small-24B-2507": "mistralai/Voxtral-Small-24B-2507"
        }
        
        new_model_path = model_mapping.get(model_name, "mistralai/Voxtral-Mini-3B-2507")
        
        if self.model_name != new_model_path:
            print(f"🔄 Switching to {model_name}")
            self.model_name = new_model_path
            self.current_model_key = model_name
            # Clear current model to force reload
            self.cleanup_model()
        else:
            print(f"✅ Already using {model_name}")
    
    @gpu_model_loading(duration=120)
    def _load_model_if_needed(self):
        """Load model and processor with GPU allocation if not already loaded."""
        if self.model is not None and self.processor is not None:
            print(f"✅ Model {self.current_model_key} already loaded")
            return
        
        device = self.gpu_manager.get_device()
        dtype = self.gpu_manager.dtype
        print(f"🔄 Loading {self.current_model_key} on {device} with {dtype}...")
        
        # Load processor and model following HuggingFace reference implementation
        print(f"📦 Loading {self.current_model_key} (original Mistral model)")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Use reference implementation from HuggingFace docs
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device
        )
        
        print(f"✅ {self.current_model_key} loaded successfully on {device}")
        
        # Print memory info if available
        if self.gpu_manager.is_gpu_available():
            memory_info = self.gpu_manager.get_memory_info()
            if memory_info["available"]:
                if memory_info["device"] == "cuda":
                    allocated_gb = memory_info["allocated"] / (1024**3)
                    print(f"📊 CUDA Memory allocated: {allocated_gb:.2f}GB")
                elif memory_info["device"] == "mps":
                    allocated_mb = memory_info["allocated"] / (1024**2)
                    print(f"📊 MPS Memory allocated: {allocated_mb:.1f}MB")
    
    def _get_audio_duration(self, wav_path: str) -> float:
        """Get audio duration in minutes."""
        audio = AudioSegment.from_file(wav_path)
        return len(audio) / (1000 * 60)
    
    def _apply_audio_trim(self, wav_path: str, start_trim: float, end_trim: float) -> str:
        """
        Apply start and end trim to audio file.
        
        Args:
            wav_path (str): Path to original audio file
            start_trim (float): Seconds to trim from beginning
            end_trim (float): Seconds to trim from end
            
        Returns:
            str: Path to trimmed audio file (or original if no trim)
        """
        if start_trim <= 0 and end_trim <= 0:
            return wav_path  # No trim needed
            
        print(f"✂️ Applying trim: {start_trim}s from start, {end_trim}s from end")
        
        audio = AudioSegment.from_file(wav_path)
        original_duration = len(audio) / 1000  # in seconds
        
        # Calculate trim positions
        start_ms = int(start_trim * 1000)
        end_ms = len(audio) - int(end_trim * 1000) if end_trim > 0 else len(audio)
        
        # Validate trim values
        if start_ms >= end_ms:
            print("⚠️ Warning: Trim values would remove entire audio, ignoring trim")
            return wav_path
            
        # Apply trim
        trimmed_audio = audio[start_ms:end_ms]
        
        # Save trimmed audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            trimmed_path = tmp_file.name
        
        trimmed_audio.export(trimmed_path, format="wav")
        
        new_duration = len(trimmed_audio) / 1000
        print(f"🎵 Audio trimmed: {original_duration:.1f}s → {new_duration:.1f}s")
        
        return trimmed_path
    
    def _create_time_chunks(self, wav_path: str) -> List[Tuple[float, float]]:
        """Create time-based chunks for processing with model-optimized durations."""
        total_duration = self._get_audio_duration(wav_path) * 60  # seconds
        # Use model-specific optimized chunk duration for Zero GPU
        chunk_minutes = self.chunk_durations.get(self.current_model_key, 15)
        max_chunk_seconds = chunk_minutes * 60
        
        print(f"🎯 Using {chunk_minutes}min chunks optimized for {self.current_model_key} on Zero GPU")
        
        if total_duration <= max_chunk_seconds:
            return [(0, total_duration)]
        
        chunks = []
        current_start = 0
        
        while current_start < total_duration:
            chunk_end = min(current_start + max_chunk_seconds, total_duration)
            chunks.append((current_start, chunk_end))
            current_start = chunk_end
        
        return chunks
    
    def _extract_audio_chunk(self, wav_path: str, start_time: float, end_time: float) -> str:
        """Extract audio chunk between timestamps."""
        audio = AudioSegment.from_file(wav_path)
        
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        chunk = audio[start_ms:end_ms]
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            chunk_path = tmp_chunk.name
        
        chunk.export(chunk_path, format="wav")
        return chunk_path
    
    @gpu_long_task(duration=300)
    def analyze_audio_chunks(
        self, 
        wav_path: str, 
        language: str = "french", 
        selected_sections: list = None,
        reference_speakers_data: str = None,
        start_trim: float = 0,
        end_trim: float = 0,
        progress_callback = None
    ) -> Dict[str, str]:
        """
        Analyze audio by chunks using Voxtral with Zero GPU.
        Uses model-optimized chunk durations (15min for Mini, 10min for Small).
        
        Args:
            wav_path (str): Path to audio file
            language (str): Expected language
            selected_sections (list): Analysis sections to include
            reference_speakers_data (str): Speaker diarization data
            start_trim (float): Seconds to trim from the beginning (default: 0)
            end_trim (float): Seconds to trim from the end (default: 0)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dict[str, str]: Analysis results
        """
        try:
            # Ensure model is loaded
            self._load_model_if_needed()
            
            total_start_time = time.time()
            
            # Apply audio trim if specified
            processed_wav_path = self._apply_audio_trim(wav_path, start_trim, end_trim)
            cleanup_trimmed_file = processed_wav_path != wav_path  # Track if we need to cleanup
            
            duration = self._get_audio_duration(processed_wav_path)
            print(f"🎵 Processing audio duration: {duration:.1f} minutes")
            
            # Create chunks with model-optimized duration
            chunks = self._create_time_chunks(processed_wav_path)
            chunk_minutes = self.chunk_durations.get(self.current_model_key, 15)
            print(f"📦 Splitting into {len(chunks)} chunks of {chunk_minutes}min")
            
            # Calculate total steps for progress (chunks + synthesis if multiple chunks)
            total_steps = len(chunks) + (1 if len(chunks) > 1 else 0)
            
            chunk_summaries = []
            
            for i, (start_time, end_time) in enumerate(chunks):
                print(f"🎯 Processing chunk {i+1}/{len(chunks)} ({start_time/60:.1f}-{end_time/60:.1f}min)")
                
                # Update progress
                if progress_callback:
                    progress_callback((i / total_steps), f"Analyzing chunk {i+1}/{len(chunks)}")
                
                chunk_start_time = time.time()
                chunk_path = self._extract_audio_chunk(processed_wav_path, start_time, end_time)
                
                try:
                    # Analyze chunk with Zero GPU
                    chunk_summary = self._analyze_single_chunk(
                        chunk_path, 
                        selected_sections, 
                        reference_speakers_data,
                        i + 1,
                        len(chunks),
                        start_time,
                        end_time
                    )
                    
                    chunk_summaries.append(f"## Segment {i+1} ({start_time/60:.1f}-{end_time/60:.1f}min)\n\n{chunk_summary}")
                    
                    chunk_duration = time.time() - chunk_start_time
                    print(f"✅ Chunk {i+1} analyzed in {chunk_duration:.1f}s")
                    
                except Exception as e:
                    print(f"❌ Error processing chunk {i+1}: {e}")
                    chunk_summaries.append(f"**Segment {i+1}:** Processing error")
                finally:
                    # Clean up chunk file
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    
                    # GPU cleanup after each chunk
                    self.gpu_manager.cleanup_gpu()
            
            # Final synthesis if multiple chunks
            if len(chunk_summaries) > 1:
                print(f"🔄 Final synthesis of {len(chunk_summaries)} segments...")
                
                # Update progress for synthesis
                if progress_callback:
                    progress_callback((len(chunks) / total_steps), "Final synthesis in progress...")
                
                combined_content = "\n\n".join(chunk_summaries)
                final_analysis = self._synthesize_chunks_final(combined_content, selected_sections)
                
                # Complete progress
                if progress_callback:
                    progress_callback(1.0, "Analysis completed!")
            else:
                # Complete progress for single chunk
                if progress_callback:
                    progress_callback(1.0, "Analysis completed!")
                final_analysis = chunk_summaries[0] if chunk_summaries else "No analysis available."
            
            total_duration = time.time() - total_start_time
            print(f"⏱️ Total analysis completed in {total_duration:.1f}s for {duration:.1f}min of audio")
            
            # Print token usage
            self.token_tracker.print_summary()
            
            return {"transcription": final_analysis}
            
        finally:
            # Cleanup trimmed audio file if created
            if cleanup_trimmed_file and os.path.exists(processed_wav_path):
                os.remove(processed_wav_path)
                print("🧹 Trimmed audio file cleaned up")
            
            # Final GPU cleanup
            self.gpu_manager.cleanup_gpu()
    
    @gpu_inference(duration=120)
    def _analyze_single_chunk(
        self, 
        chunk_path: str, 
        selected_sections: list,
        reference_speakers_data: str,
        chunk_num: int,
        total_chunks: int,
        start_time: float,
        end_time: float
    ) -> str:
        """Analyze a single audio chunk with GPU inference."""
        # Build analysis prompt
        sections_list = selected_sections if selected_sections else ["resume_executif"]
        chunk_info = f"SEGMENT {chunk_num}/{total_chunks} ({start_time/60:.1f}-{end_time/60:.1f}min)" if total_chunks > 1 else None
        
        prompt_text = VoxtralPrompts.get_meeting_summary_prompt(
            sections_list, 
            reference_speakers_data, 
            chunk_info, 
            None
        )
        
        # Debug: Print prompt to check language
        print("🔍 DEBUG PROMPT:")
        print("=" * 50)
        print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)
        print("=" * 50)
        
        # Create conversation for audio instruct mode
        conversation = [{
            "role": "user", 
            "content": [
                {"type": "audio", "path": chunk_path},
                {"type": "text", "text": prompt_text},
            ],
        }]
        
        # Process with chat template
        inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
        device = self.gpu_manager.get_device()
        dtype = self.gpu_manager.dtype if hasattr(self.gpu_manager, 'dtype') else torch.float16
        
        # Move inputs to device with appropriate dtype
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device, dtype=dtype)
        else:
            # Handle BatchFeature or dict-like inputs
            inputs = {k: v.to(device, dtype=dtype) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate with optimized settings for Spaces
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=8000,  # Optimized for pre-quantized model efficiency
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                output_scores=False
            )
        
        # Decode response
        input_tokens = inputs.input_ids.shape[1]
        output_tokens_count = outputs.shape[1] - input_tokens
        
        chunk_summary = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        # Track tokens
        self.token_tracker.add_chunk_tokens(input_tokens, output_tokens_count)
        
        return chunk_summary
    
    @gpu_inference(duration=60)
    def _synthesize_chunks_final(self, combined_content: str, selected_sections: list) -> str:
        """Final synthesis of all chunks with GPU inference."""
        try:
            # Build synthesis prompt
            sections_text = ""
            if selected_sections:
                for section_key in selected_sections:
                    if section_key in VoxtralPrompts.AVAILABLE_SECTIONS:
                        section = VoxtralPrompts.AVAILABLE_SECTIONS[section_key]
                        sections_text += f"\n{section['title']}\n{section['description']}\n"
            
            synthesis_prompt = f"""Voici des analyses détaillées de plusieurs segments de réunion :

{combined_content}

INSTRUCTION CRITIQUE - LANGUE DE RÉPONSE :
- DÉTECTE la langue utilisée dans les segments ci-dessus
- RÉPONDS OBLIGATOIREMENT dans cette même langue détectée
- Si les segments sont en français → réponds en français
- Si les segments sont en anglais → réponds en anglais

Maintenant synthétise ces analyses en un résumé global cohérent structuré selon les sections demandées :{sections_text}

Fournis une synthèse unifiée qui combine et résume les informations de tous les segments de manière cohérente."""
            
            # Generate synthesis
            conversation = [{"role": "user", "content": synthesis_prompt}]
            inputs = self.processor.apply_chat_template(conversation, return_tensors="pt")
            device = self.gpu_manager.get_device()
            dtype = self.gpu_manager.dtype if hasattr(self.gpu_manager, 'dtype') else torch.float16
            
            # Move inputs to device with appropriate dtype
            if hasattr(inputs, 'to'):
                inputs = inputs.to(device, dtype=dtype)
            else:
                inputs = {k: v.to(device, dtype=dtype) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=3000,  # Optimized for pre-quantized efficiency
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode synthesis
            input_length = inputs.input_ids.shape[1]
            output_tokens_count = outputs.shape[1] - input_length
            
            final_synthesis = self.processor.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            self.token_tracker.add_synthesis_tokens(input_length, output_tokens_count)
            
            return f"# Global Meeting Summary\n\n{final_synthesis}\n\n---\n\n## Details by Segment\n\n{combined_content}"
            
        except Exception as e:
            print(f"❌ Error during final synthesis: {e}")
            return f"# Meeting Summary\n\n⚠️ Error during final synthesis: {str(e)}\n\n## Segment Analyses\n\n{combined_content}"
    
    def cleanup_model(self):
        """Clean up model from memory."""
        if self.model is not None:
            self.model.to('cpu')
            del self.model
            self.model = None
            
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        self.gpu_manager.cleanup_gpu()
        print("🧹 Voxtral Spaces model cleaned up")