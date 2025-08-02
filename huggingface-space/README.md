---
title: 🎙️ MeetingNotes - AI Analysis with Voxtral
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
hardware: zerogpu
short_description: AI meeting analysis with Voxtral transcription & summary
models:
- mistralai/Voxtral-Mini-3B-2507
- mistralai/Voxtral-Small-24B-2507
tags:
- meeting-analysis
- transcription
- voxtral
- ai-summarization
- zero-gpu
- mcp-server
- progress-tracking
---

# 🎙️ MeetingNotes - AI Analysis with Voxtral

Transform your meeting recordings into structured, actionable insights with advanced AI analysis powered by standard Mistral Voxtral models optimized for Zero GPU.

## ⚡ Features

🤖 **Advanced AI Models**:
- **Voxtral-Mini-3B-2507**: Fast processing for quick analysis (15min chunks)
- **Voxtral-Small-24B-2507**: Enhanced quality for detailed summaries (10min chunks)
- **Standard Mistral Models**: Original models optimized for Zero GPU

🎯 **Smart Analysis**:
- **Executive Summaries**: Key decisions and outcomes
- **Action Plans**: Clear next steps and responsibilities  
- **Multi-language Support**: Automatic language detection
- **Customizable Sections**: Choose analysis focus areas
- **Progress Tracking**: Real-time progress with chunk-based updates

📊 **Processing Options**:
- **Auto-optimized Chunks**: Model-specific duration optimization for Zero GPU
- **Audio Trimming**: Process specific time ranges
- **Quality Presets**: Action-oriented vs Information meetings
- **Synthesis**: Intelligent combination of multi-chunk analysis

🔧 **Native MCP Support**:
- **analyze_meeting_audio()**: Full meeting analysis with configurable options
- **get_available_sections()**: List all available analysis sections
- **get_meeting_templates()**: Pre-configured meeting types

## 🚀 How to Use

### Basic Analysis
1. **Upload Audio**: Support for MP3, WAV, M4A, OGG formats
2. **Select Model**: Choose between Mini (fast) or Small (detailed) 
3. **Configure Sections**: Pick analysis areas (summary, actions, decisions)
4. **Analyze**: Get structured meeting insights

### Advanced Features
- **Custom Sections**: Select specific analysis focus areas
- **Meeting Presets**: Quick setup for action vs information meetings
- **Audio Trimming**: Process only relevant portions
- **Progress Tracking**: Real-time updates during processing

### MCP Integration
This app provides native MCP (Model Context Protocol) server functionality:
- Connect your MCP client to analyze meetings programmatically
- Available tools: meeting analysis, section management, template access

## 🎯 Perfect For

- **Business Meetings**: Track decisions and action items
- **Interviews**: Structured conversation analysis  
- **Conferences**: Speaker identification and key points
- **Team Standups**: Quick summaries and next steps
- **Client Calls**: Professional meeting documentation

## 🔒 Privacy & Security

- **Zero GPU**: Efficient processing without persistent storage
- **Local Processing**: Audio analyzed securely in HF Spaces environment
- **No Data Retention**: Recordings processed temporarily only

## 🔗 Related Links

- **Full Version**: Complete MeetingNotes with MLX, API modes, and speaker diarization available on [GitHub](https://github.com/VincentGourbin/meetingnotes)
- **Models**: [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | [Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)

Start analyzing your meetings with AI precision! 🎙️✨
