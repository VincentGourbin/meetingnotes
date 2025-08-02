"""AI modules for HF Spaces version."""

from .voxtral_spaces_analyzer import VoxtralSpacesAnalyzer
from .diarization import SpeakerDiarization
from .prompts_config import VoxtralPrompts

__all__ = ['VoxtralSpacesAnalyzer', 'SpeakerDiarization', 'VoxtralPrompts']