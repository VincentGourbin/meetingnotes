"""
Token usage tracking utility for MeetingNotes HF Spaces.

This module provides a centralized way to track and report token consumption
for Transformers-based processing in HF Spaces environment.
"""


class TokenTracker:
    """
    Centralized token usage tracking for HF Spaces.
    
    Tracks input and output tokens across different chunks and processing modes
    to provide comprehensive usage statistics.
    """
    
    def __init__(self, mode: str = "Transformers-8bit"):
        self.mode = mode
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.chunks_processed = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.synthesis_input_tokens = 0
        self.synthesis_output_tokens = 0
    
    def set_mode(self, mode: str):
        """Set the processing mode for reporting."""
        self.mode = mode
    
    def add_chunk_tokens(self, input_tokens: int, output_tokens: int):
        """Add tokens from a chunk processing."""
        self.chunks_processed += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        print(f"📊 Stats {self.mode} Chunk {self.chunks_processed} - Input: {input_tokens} tokens, Output: {output_tokens} tokens")
    
    def add_synthesis_tokens(self, input_tokens: int, output_tokens: int):
        """Add tokens from synthesis processing."""
        self.synthesis_input_tokens = input_tokens
        self.synthesis_output_tokens = output_tokens
        
        print(f"📊 Stats {self.mode} Synthesis - Input: {input_tokens} tokens, Output: {output_tokens} tokens")
    
    def print_summary(self):
        """Print final token usage summary."""
        total_input = self.total_input_tokens + self.synthesis_input_tokens
        total_output = self.total_output_tokens + self.synthesis_output_tokens
        grand_total = total_input + total_output
        
        print(f"\n📊 === TOKEN USAGE SUMMARY ({self.mode}) ===")
        print(f"📦 Chunks processed: {self.chunks_processed}")
        print(f"📥 Total input tokens: {total_input:,}")
        print(f"📤 Total output tokens: {total_output:,}")
        print(f"🔢 Grand total: {grand_total:,} tokens")
        
        if self.synthesis_input_tokens > 0:
            print(f"   • Chunk analysis: {self.total_input_tokens + self.total_output_tokens:,} tokens")
            print(f"   • Final synthesis: {self.synthesis_input_tokens + self.synthesis_output_tokens:,} tokens")
        
        print("=" * 50)