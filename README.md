# MeetingNotes - Intelligent Meeting Analysis with Voxtral

Web application using **Voxtral AI from Mistral AI** to automatically analyze your audio/video meetings with:
- **Direct analysis**: Transcription and structured summary in one step
- **3 processing modes**: Local (Transformers), MLX (Apple Silicon), API (Cloud)
- **Quantized models**: 4bit/8bit support for memory efficiency
- **Smart diarization**: Speaker identification and renaming
- **Customizable summaries**: Modular sections according to your needs
- **Language-adaptive**: Automatically detects and responds in meeting language
- **Centralized UI**: Clean English interface with multilingual analysis

![Meeting Analysis Interface](Assets/meeting%20analysis%20parameters.png)

## 🚀 Quick Installation

1. **Clone the repository and install dependencies:**
```bash
git clone <repository-url>
cd meetingnotes
pip install -r requirements.txt
```

2. **Configure your Hugging Face token:**
```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
```

3. **Launch the application:**
```bash
python main.py
```

The web interface will be accessible at **http://localhost:7860**

## ⚙️ Configuration

### Hugging Face Token (Required)
Get an access token from [Hugging Face](https://huggingface.co/settings/tokens) and add it to `.env`:
```env
HUGGINGFACE_TOKEN=your_token_here
```

### Mistral API Key (Optional)
To use cloud API mode, get a key from [Mistral AI](https://console.mistral.ai/):
```env
MISTRAL_API_KEY=your_mistral_api_key
```

## 🎯 Features

### Processing Modes

Choose the mode that best fits your hardware and needs:

#### Local Mode (Transformers)
![Local Mode](Assets/local%20mode.png)
- **Local processing**: Everything runs on your machine with PyTorch
- **Privacy**: No data sent to external servers
- **GPU acceleration**: Automatic CUDA/MPS detection

#### MLX Mode (Apple Silicon)
![MLX Mode](Assets/mlx%20mode.png)
- **Optimized for Mac**: M1/M2/M3 processors with MLX Framework
- **Best performance**: Native Apple Silicon acceleration
- **Memory efficient**: Optimized quantized models

#### API Mode (Cloud)
![API Mode](Assets/api%20mode.png)
- **Cloud processing**: Uses Mistral Cloud API
- **No local resources**: Minimal memory usage
- **Always up-to-date**: Latest models and improvements

### Models and Quantization
| Model | Precision | Repository | Memory Usage |
|-------|-----------|------------|--------------|
| **Voxtral Mini** | Default | `mistralai/Voxtral-Mini-3B-2507` | ~6GB |
| **Voxtral Mini** | 8bit | `mzbac/voxtral-mini-3b-8bit` | ~3.5GB |
| **Voxtral Mini** | 4bit | `mzbac/voxtral-mini-3b-4bit-mixed` | ~2GB |
| **Voxtral Small** | Default | `mistralai/Voxtral-Small-24B-2507` | ~48GB |
| **Voxtral Small** | 8bit | `VincentGOURBIN/voxtral-small-8bit` | ~24GB |
| **Voxtral Small** | 4bit | `VincentGOURBIN/voxtral-small-4bit-mixed` | ~12GB |

### Speaker Diarization

![Speaker Diarization](Assets/diarization.png)

- **Automatic identification**: Detection of different speakers with pyannote.audio
- **Reference segments**: Listen to audio samples for each speaker
- **Custom renaming**: Assign human names to speakers
- **Context integration**: Use speaker information in summaries

### Customizable Summaries
**Modular sections**: Choose the sections to include according to your needs
- **📄 Executive Summary**: Global overview of the meeting
- **💬 Main Discussions**: Main topics addressed
- **✅ Action Plan**: Actions, responsibilities, deadlines
- **⚖️ Decisions Made**: Validated decisions
- **⏭️ Next Steps**: Follow-up actions
- **📌 Main Topics**: Information presented
- **⭐ Key Points**: Insights and key data
- **❓ Questions & Discussions**: Questions asked and answers
- **📝 Follow-up Elements**: Clarifications needed

**Predefined Profiles**:
- **🎯 Action Profile**: Focus on tasks and decisions
- **📊 Information Profile**: Focus on data and insights
- **📋 Complete Profile**: All sections activated

### Supported Formats
- **Audio**: WAV, MP3, M4A, OGG, FLAC
- **Video**: MP4, AVI, MOV, MKV (automatic audio extraction)

## 🔧 Usage

### 1. Processing Mode Configuration
1. **Choose the mode**: Local, MLX or API
2. **Select the model**: Mini or Small according to your needs
3. **Choose precision**: Default, 8bit or 4bit to optimize memory

### 2. Upload and Options
- **File**: Direct audio or video (automatic extraction)
- **Optional trimming**: Start/end trimming (leave empty for 0)

![Trim Options](Assets/trim_options.png)

- **Chunk size**: Processing duration (5-25 minutes)

### 3. Diarization (Optional)
1. **Analyze speakers** with pyannote.audio
2. **Listen to reference segments** of each speaker
3. **Rename speakers** with custom names
4. **Apply renamings** for enriched context

### 4. Summary Customization
- **Modular sections**: Enable only necessary sections
- **Quick profiles**: Action, Information or Complete
- **Flexible configuration**: Adapt summary to your usage

### 5. Analysis and Results
Click **"Analyze Meeting"** to get a customized structured summary.

![Meeting Summary](Assets/meeting%20summary.png)

## 🏗️ Architecture

The project follows a modular architecture in `src/meetingnotes/`:

```
src/meetingnotes/
├── ai/                    # Artificial Intelligence
│   ├── voxtral_analyzer.py      # Local Voxtral analyzer (Transformers)
│   ├── voxtral_api_analyzer.py  # Voxtral API analyzer
│   ├── voxtral_mlx_analyzer.py  # Voxtral MLX analyzer (Apple Silicon)
│   ├── diarization.py           # Speaker diarization (pyannote)
│   ├── memory_manager.py        # Optimized memory management
│   └── prompts_config.py        # Centralized prompt configuration
├── audio/                 # Audio Processing
│   ├── wav_converter.py         # Format conversion
│   └── normalizer.py            # Volume normalization
├── core/                  # Business Logic
│   ├── voxtral_direct.py        # Direct processing (Transformers)
│   ├── voxtral_api.py           # Mistral API interface
│   └── voxtral_mlx.py           # MLX Apple Silicon interface
├── ui/                    # User Interface
│   ├── main.py                  # Main Gradio interface
│   ├── handlers.py              # Event handlers
│   └── labels.py                # UI labels and text constants
└── utils/                 # Utilities
    ├── __init__.py              # Utils module
    ├── time_formatter.py        # Duration formatting
    └── token_tracker.py          # Token usage tracking
```

For more details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🔧 Advanced Configuration

### Environment Variables
```env
# Required for all modes
HUGGINGFACE_TOKEN=your_hf_token

# Optional for API mode
MISTRAL_API_KEY=your_mistral_key
```

### Hardware Optimizations
- **Mac M1/M2/M3**: Use MLX mode for better performance
- **NVIDIA GPU**: Local mode with automatic CUDA acceleration
- **CPU only**: Prefer 4bit models to save memory
- **Limited memory**: Mini 4bit (~2GB) or Small 4bit (~12GB)

## 🔍 Technical Features

### Memory Optimizations
- **Pre-quantized models**: 4bit and 8bit for memory reduction
- **Memory manager**: Automatic cleanup between chunks
- **Multi-platform support**: MPS (Apple), CUDA (NVIDIA), optimized CPU

### Intelligent Processing
- **3 inference modes**: Direct audio-chat without intermediate transcription  
- **Language-adaptive**: Automatically detects meeting language and responds accordingly
- **Adaptive chunks**: Smart division of long files with synthesis
- **Modular prompts**: Customizable summary sections with centralized configuration
- **Enriched context**: Integration of diarization in analyses
- **Token tracking**: Comprehensive usage statistics across all processing modes

### Modern Interface
- **Centralized UI labels**: Clean English interface with maintainable text management
- **Improved API layout**: API key positioned next to model selection for better UX
- **Interactive diarization**: Speaker listening and renaming
- **Modular sections**: Advanced summary customization with preset profiles
- **Real-time feedback**: Detailed progress indicators and token consumption tracking

## 📦 Main Dependencies

- **gradio**: Modern web user interface
- **torch/torchaudio**: Deep learning framework (Local mode)
- **transformers**: Hugging Face and Voxtral models
- **mlx/mlx-voxtral**: MLX framework optimized for Apple Silicon (macOS only)
- **pyannote.audio**: Speaker diarization
- **pydub**: Audio processing and conversion
- **requests**: Communication with Mistral API
- **python-dotenv**: Environment variables management

## 🔒 Security and Privacy

- **Local processing**: Option for entirely on-machine processing
- **Environment variables**: Secure tokens via `.env`
- **No cloud storage**: Your files remain local
- **Automatic cleanup**: Temporary files removal

## 🚦 Project Status

✅ **Version v2.1** - Enhanced User Experience & Language Support
- **3 processing modes**: Local, MLX, API with improved layout
- **6 model configurations**: Mini/Small + Default/8bit/4bit  
- **Complete diarization**: Speaker identification and renaming
- **Modular summaries**: 9 customizable sections with preset profiles
- **Language-adaptive AI**: Automatically responds in detected meeting language
- **Centralized UI management**: Clean English interface with maintainable labels
- **Token tracking**: Comprehensive usage statistics for all modes
- **Improved UX**: Better API mode layout and visual organization
- **Multi-platform support**: Windows, macOS, Linux

## 🤝 Contributing

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if necessary
5. Open a Pull Request

## 📄 License

This project is under MIT license. See the LICENSE file for more details.

---

**MeetingNotes** - Powered by [Voxtral from Mistral AI](https://mistral.ai/) | 🚀 Intelligent meeting analysis | 💾 Secure local and cloud processing