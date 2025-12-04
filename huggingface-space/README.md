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
- pyannote/speaker-diarization-3.1
tags:
- meeting-analysis
- transcription
- voxtral
- speaker-diarization
- ai-summarization
- zero-gpu
- mcp-server
---

# 🎙️ MeetingNotes - AI Analysis with Voxtral

Transform your meeting recordings into structured, actionable insights with advanced AI analysis powered by Voxtral models and speaker diarization.

## ⚡ Features

🤖 **Advanced AI Models**:
- **Voxtral-Mini-3B-2507**: Fast processing for quick analysis
- **Voxtral-Small-24B-2507**: Enhanced quality for detailed summaries
- **8-bit quantization**: Optimized for Zero GPU efficiency

🎯 **Smart Analysis**:
- **Executive Summaries**: Key decisions and outcomes
- **Action Plans**: Clear next steps and responsibilities  
- **Speaker Diarization**: Identify and separate different speakers
- **Multi-language Support**: Automatic language detection
- **Customizable Sections**: Choose analysis focus areas

📊 **Processing Options**:
- **Chunk-based Processing**: Handle long recordings efficiently
- **Reference Segments**: Preview identified speakers
- **Audio Trimming**: Process specific time ranges
- **Quality Presets**: Action-oriented vs Information meetings

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
- **Speaker Diarization**: Enable to identify different speakers
- **Custom Sections**: Select specific analysis focus areas
- **Meeting Presets**: Quick setup for action vs information meetings
- **Audio Trimming**: Process only relevant portions

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

Start analyzing your meetings with AI precision! 🎙️✨
