"""
UI Labels and Text Constants for MeetingNotes Application.

This module centralizes all user interface text, labels, and messages
to facilitate maintenance and potential internationalization.
"""


class UILabels:
    """Centralized UI labels and text constants."""
    
    # Main header
    MAIN_TITLE = "🎙️ MeetingNotes"
    MAIN_SUBTITLE = "### Intelligent Meeting Analysis with AI"
    MAIN_DESCRIPTION = "Powered by Voxtral"
    
    # Processing mode section
    PROCESSING_MODE_TITLE = "### 🎯 Processing Mode"
    PROCESSING_MODE_LABEL = "Processing Mode"
    PROCESSING_MODE_INFO = "Local: Transformers | MLX: Apple Silicon optimized | API: Mistral Cloud"
    
    # Model selection
    LOCAL_MODEL_LABEL = "🤖 Local Model"
    LOCAL_MODEL_INFO = "Mini: Faster | Small: More accurate, more memory"
    MLX_MODEL_LABEL = "🚀 MLX Model"  
    MLX_MODEL_INFO = "Mini: Faster | Small: More accurate, more memory"
    API_MODEL_LABEL = "🌐 API Model"
    API_MODEL_INFO = "Mini: Faster, cheaper | Small: More accurate, more expensive"
    
    # Precision selection
    LOCAL_PRECISION_LABEL = "⚡ Local Precision"
    LOCAL_PRECISION_INFO = "Default: Max quality | 8bit: Good compromise | 4bit: Memory saving"
    MLX_PRECISION_LABEL = "⚡ MLX Precision"
    MLX_PRECISION_INFO = "Default: Max quality | 8bit: Good compromise | 4bit: Memory saving"
    
    # API Key
    API_KEY_LABEL = "🔑 Mistral API Key"
    API_KEY_PLACEHOLDER = "Enter your Mistral API key..."
    API_KEY_INFO = "Required to use Mistral API"
    
    # Input mode
    INPUT_MODE_TITLE = "### 📝 Input Mode"
    INPUT_MODE_LABEL = "File Type"
    INPUT_MODE_INFO = "Audio: for .wav, .mp3, etc. | Video: for .mp4, .avi, etc."
    INPUT_MODE_AUDIO = "🎵 Audio"
    INPUT_MODE_VIDEO = "🎬 Video"
    
    # Audio section
    AUDIO_MODE_TITLE = "### 🎵 Audio Mode"
    AUDIO_INPUT_LABEL = "🎙️ Recording or audio file"
    
    # Video section  
    VIDEO_MODE_TITLE = "### 🎬 Video Mode"
    VIDEO_INPUT_LABEL = "📁 Video file"
    EXTRACT_AUDIO_BUTTON = "🔄 Extract audio and switch to Audio mode"
    
    # Trim options
    TRIM_OPTIONS_TITLE = "✂️ Trim Options (optional)"
    START_TRIM_LABEL = "⏪ Remove X seconds from start"
    START_TRIM_INFO = "Number of seconds to remove from start of file"
    END_TRIM_LABEL = "⏩ Remove X seconds from end"
    END_TRIM_INFO = "Number of seconds to remove from end of file"
    
    # Diarization
    DIARIZATION_TITLE = "👥 Speaker Identification (optional)"
    DIARIZATION_DESCRIPTION = "🔍 **Automatic diarization**: Analyze different speakers present in audio with pyannote."
    NUM_SPEAKERS_LABEL = "👤 Number of speakers (optional)"
    NUM_SPEAKERS_INFO = "Leave empty for automatic detection"
    NUM_SPEAKERS_PLACEHOLDER = "Auto"
    DIARIZE_BUTTON = "🎤 Analyze speakers"
    
    # Reference segments
    REFERENCE_SEGMENTS_TITLE = "### 🎵 Reference Segments"
    REFERENCE_SEGMENTS_DESCRIPTION = "Click on a speaker to listen to their reference segment:"
    SPEAKERS_DETECTED_LABEL = "👥 Detected speakers"
    SPEAKERS_DETECTED_INFO = "Select a speaker to listen to their segment"
    REFERENCE_AUDIO_LABEL = "🔊 Reference segment"
    
    # Speaker renaming
    SPEAKER_RENAME_TITLE = "### ✏️ Rename Speaker"
    SPEAKER_NAME_LABEL = "📝 New name"
    SPEAKER_NAME_PLACEHOLDER = "Enter speaker name (e.g. John, Mary...)"
    SPEAKER_NAME_INFO = "The name will replace the selected speaker ID"
    APPLY_RENAME_BUTTON = "✅ Apply all renamings"
    IDENTIFIED_SPEAKERS_LABEL = "👥 Identified speakers"
    IDENTIFIED_SPEAKERS_INFO = "List of detected speakers with their custom names"
    
    # Main analysis
    MAIN_ANALYSIS_TITLE = "### ⚡ Meeting Analysis"
    MAIN_ANALYSIS_DESCRIPTION = "💡 **Voxtral AI**: Smart transcription and structured summary of your meeting."
    CHUNK_DURATION_LABEL = "📦 Chunk size (minutes)"
    CHUNK_DURATION_INFO = "Duration of each audio chunk to process separately"
    
    # Summary sections
    SUMMARY_SECTIONS_TITLE = "### 📋 Summary Sections"
    SUMMARY_SECTIONS_DESCRIPTION = "Customize the sections to include in your summary:"
    
    # Preset buttons
    PRESET_ACTION_BUTTON = "🎯 Action Profile"
    PRESET_INFO_BUTTON = "📊 Information Profile"
    PRESET_COMPLETE_BUTTON = "📋 Complete Profile"
    
    # Section categories
    ACTION_SECTIONS_TITLE = "**🎯 Action-oriented sections**"
    INFO_SECTIONS_TITLE = "**📊 Information-oriented sections**"
    
    # Individual sections
    SECTION_EXECUTIVE_SUMMARY = "📄 Executive Summary"
    SECTION_EXECUTIVE_SUMMARY_INFO = "Global overview of the meeting"
    SECTION_MAIN_DISCUSSIONS = "💬 Main Discussions"
    SECTION_MAIN_DISCUSSIONS_INFO = "Main topics addressed"
    SECTION_ACTION_PLAN = "✅ Action Plan"
    SECTION_ACTION_PLAN_INFO = "Actions, responsibilities, deadlines"
    SECTION_DECISIONS = "⚖️ Decisions Made"
    SECTION_DECISIONS_INFO = "Validated decisions"
    SECTION_NEXT_STEPS = "⏭️ Next Steps"
    SECTION_NEXT_STEPS_INFO = "Follow-up actions"
    SECTION_MAIN_TOPICS = "📌 Main Topics"
    SECTION_MAIN_TOPICS_INFO = "Information presented"
    SECTION_KEY_POINTS = "⭐ Key Points"
    SECTION_KEY_POINTS_INFO = "Insights and key data"
    SECTION_QUESTIONS = "❓ Questions & Discussions"
    SECTION_QUESTIONS_INFO = "Questions asked and answers"
    SECTION_FOLLOW_UP = "📝 Follow-up Elements"
    SECTION_FOLLOW_UP_INFO = "Clarifications needed"
    
    # Analysis button
    ANALYZE_BUTTON = "⚡ Analyze Meeting"
    
    # Results
    RESULTS_TITLE = "### 📋 Meeting Summary"
    RESULTS_LABEL = "📄 Structured Meeting Summary"
    RESULTS_PLACEHOLDER = "The structured summary will appear here after analysis..."
    
    # Footer
    FOOTER_TEXT = """
    ---
    **MeetingNotes** | Powered by [Voxtral](https://mistral.ai/) | 
    🚀 Intelligent meeting analysis | 💾 Secure local and cloud processing
    """
    
    # Error messages
    ERROR_NO_AUDIO_FILE = "❌ No audio file provided."
    ERROR_NO_HF_TOKEN = "❌ Hugging Face token required for diarization."
    ERROR_NO_API_KEY = "❌ Mistral API key required for API mode."
    ERROR_AUDIO_PROCESSING = "❌ Error processing audio file."
    ERROR_DIARIZATION = "❌ Error during diarization:"
    ERROR_AUDIO_EXTRACTION = "❌ Error during audio extraction:"
    ERROR_SPEAKER_SELECTION = "❌ Error during speaker selection:"
    ERROR_ANALYSIS = "❌ Error during analysis:"
    
    # Success messages
    SUCCESS_ANALYSIS_COMPLETE = "✅ Analysis completed successfully"
    SUCCESS_MODEL_LOADED = "✅ Model loaded successfully"
    SUCCESS_DIARIZATION_COMPLETE = "✅ Diarization completed"
    
    # Info messages
    INFO_LOADING_MODEL = "🔄 Loading model..."
    INFO_PROCESSING_AUDIO = "🎵 Processing audio..."
    INFO_ANALYZING_SPEAKERS = "🎤 Analyzing speakers..."
    INFO_GENERATING_SUMMARY = "📝 Generating summary..."
    INFO_CHUNK_PROCESSING = "🎯 Processing chunk"
    INFO_SYNTHESIS = "🔄 Final synthesis..."
    
    # Models and choices
    MODEL_MINI = "Voxtral-Mini-3B-2507"
    MODEL_SMALL = "Voxtral-Small-24B-2507"
    PRECISION_DEFAULT = "Default"
    PRECISION_8BIT = "8bit"
    PRECISION_4BIT = "4bit"
    MODE_LOCAL = "Local"
    MODE_MLX = "MLX"
    MODE_API = "API"
    
    # API models
    API_MODEL_MINI = "voxtral-mini-latest"
    API_MODEL_SMALL = "voxtral-small-latest"


class LogMessages:
    """Log messages and console output text."""
    
    # Processing
    PROCESSING_START = "🚀 === Direct audio analysis ==="
    PROCESSING_FILE = "📂 File:"
    PROCESSING_MODEL = "🤖 Model:"
    PROCESSING_LANGUAGE = "🌍 Language:"
    PROCESSING_SECTIONS = "📊 Sections:"
    PROCESSING_CHUNK_DURATION = "⏱️ Chunk duration:"
    
    # Model loading
    MODEL_LOADING = "🔄 Loading model..."
    MODEL_LOADED = "✅ Model loaded successfully"
    MODEL_CLEANUP = "🧹 Cleaning up model..."
    MODEL_CLEANED = "✅ Model cleaned"
    
    # Memory management
    MEMORY_STATS = "📊 Memory stats:"
    MEMORY_CLEANUP = "🧹 Memory cleanup..."
    MEMORY_FULL_CLEANUP = "🧹 Full memory cleanup"
    
    # Analysis
    ANALYSIS_START = "🔄 Starting analysis"
    ANALYSIS_CHUNK = "🎯 Processing chunk"
    ANALYSIS_COMPLETE = "✅ Analysis completed in"
    ANALYSIS_ERROR = "❌ Analysis error:"
    
    # Diarization
    DIARIZATION_START = "🎤 Starting diarization:"
    DIARIZATION_SPEAKERS_FOUND = "👥 Speakers found:"
    DIARIZATION_SEGMENTS = "📋 Segments created:"
    DIARIZATION_RENAME = "💾 Speaker rename:"
    
    # Audio processing  
    AUDIO_DURATION = "🎵 Audio duration:"
    AUDIO_CHUNK_EXTRACT = "📦 Extracting chunk:"
    AUDIO_FILE_SHORT = "📄 Short file, single chunk processing"
    AUDIO_CHUNKS_CREATED = "📦 Split into chunks:"
    
    # Synthesis
    SYNTHESIS_START = "🔄 Final synthesis of segments..."
    SYNTHESIS_COMPLETE = "✅ Analysis with final synthesis completed"
    SYNTHESIS_ERROR = "❌ Error during final synthesis:"