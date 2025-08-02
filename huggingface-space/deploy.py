#!/usr/bin/env python3
"""
HuggingFace Spaces Deployment Script for MeetingNotes
Deploys the MeetingNotes app to HuggingFace Spaces with ZeroGPU support and native MCP
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import tempfile
import getpass

# Configuration
SPACE_NAME = "VincentGOURBIN/MeetingNotes-Voxtral-Analysis"
SPACE_TITLE = "🎙️ MeetingNotes - AI Analysis with Voxtral"
SPACE_DESCRIPTION = "AI meeting analysis with Voxtral transcription & summary"

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")
    
    # Check git
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ Git is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed. Please install git first.")
        sys.exit(1)
    
    # Check huggingface_hub
    try:
        import huggingface_hub
        print("✅ HuggingFace Hub is available")
    except ImportError:
        print("❌ HuggingFace Hub not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        print("✅ HuggingFace Hub installed")

def get_hf_token():
    """Get HuggingFace token from environment or user input"""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if not token:
        print("\n🔑 HuggingFace token required for deployment")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        print("Make sure your token has 'write' permissions")
        token = getpass.getpass("Enter your HuggingFace token: ").strip()
    
    if not token:
        print("❌ No token provided. Deployment cancelled.")
        sys.exit(1)
    
    return token

def create_space_config():
    """Create the Space configuration files"""
    print("📝 Creating Space configuration...")
    
    # Create README.md for the space
    readme_content = f"""---
title: {SPACE_TITLE}
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
hardware: zerogpu
short_description: {SPACE_DESCRIPTION}
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
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create requirements.txt specifically for Spaces
    requirements_content = """gradio[mcp]>=5.34.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
mistral-common[audio]>=1.8.0
soundfile>=0.12.1
librosa>=0.10.0
pydub>=0.25.0
pyannote.audio>=3.1.0
speechbrain>=0.5.0
python-dotenv>=1.0.0
httpx>=0.24.0
numpy>=1.21.0
scipy>=1.9.0
safetensors>=0.3.0
tokenizers>=0.15.0
sentencepiece>=0.1.97
spaces
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Space configuration created")

def validate_files():
    """Validate that all required files exist"""
    print("🔍 Validating files...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Check if app.py has ZeroGPU decorators
    with open("app.py", "r") as f:
        app_content = f.read()
        if "@gpu_inference" not in app_content and "@spaces.GPU" not in app_content:
            print("⚠️ Warning: No GPU decorators found in app.py")
            print("Make sure your GPU-intensive functions have @gpu_inference decorators")
        else:
            print("✅ GPU decorators found")
    
    # Check MCP support
    if "mcp_server=True" not in app_content:
        print("⚠️ Warning: MCP server not enabled in app.py")
    else:
        print("✅ MCP server support enabled")
    
    print("✅ All required files present")

def validate_source_structure():
    """Validate the source code structure"""
    print("🔍 Validating source structure...")
    
    required_dirs = [
        "src/",
        "src/ai/",
        "src/ui/",
        "src/utils/"
    ]
    
    required_source_files = [
        "src/ai/voxtral_spaces_analyzer.py",
        "src/ai/diarization.py", 
        "src/ui/spaces_interface.py",
        "src/utils/zero_gpu_manager.py"
    ]
    
    missing_items = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_items.append(dir_path)
    
    for file_path in required_source_files:
        if not os.path.exists(file_path):
            missing_items.append(file_path)
    
    if missing_items:
        print(f"❌ Missing required source items: {', '.join(missing_items)}")
        sys.exit(1)
    
    print("✅ Source structure validated")

def create_space(token):
    """Create or update the HuggingFace Space"""
    print(f"🚀 Creating/updating Space: {SPACE_NAME}")
    
    from huggingface_hub import HfApi, login
    
    # Login to HuggingFace
    login(token=token, add_to_git_credential=True)
    
    api = HfApi()
    
    try:
        # Try to get space info (check if it exists)
        space_info = api.space_info(repo_id=SPACE_NAME)
        print(f"✅ Space {SPACE_NAME} already exists, updating...")
        update_mode = True
    except Exception:
        print(f"📦 Creating new Space: {SPACE_NAME}")
        update_mode = False
        
        # Create the space
        try:
            api.create_repo(
                repo_id=SPACE_NAME,
                repo_type="space",
                space_sdk="gradio",
                space_hardware="zerogpu",
                private=False
            )
            print("✅ Space created successfully")
        except Exception as e:
            print(f"❌ Failed to create space: {e}")
            sys.exit(1)
    
    return update_mode

def deploy_files(token):
    """Deploy files to the Space"""
    print("📤 Uploading files to Space...")
    
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    # Files to upload
    files_to_upload = [
        "app.py",
        "requirements.txt", 
        "README.md"
    ]
    
    # Directories to upload recursively
    dirs_to_upload = [
        "src/"
    ]
    
    try:
        # Upload individual files
        for file in files_to_upload:
            if os.path.exists(file):
                print(f"  📄 Uploading {file}...")
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=SPACE_NAME,
                    repo_type="space",
                    token=token
                )
            else:
                print(f"  ⚠️ Skipping missing file: {file}")
        
        # Upload directories
        for dir_path in dirs_to_upload:
            if os.path.exists(dir_path):
                print(f"  📁 Uploading directory {dir_path}...")
                api.upload_folder(
                    folder_path=dir_path,
                    path_in_repo=dir_path,
                    repo_id=SPACE_NAME,
                    repo_type="space",
                    token=token,
                    ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"]
                )
            else:
                print(f"  ⚠️ Skipping missing directory: {dir_path}")
        
        print("✅ All files uploaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to upload files: {e}")
        sys.exit(1)

def wait_for_space_build():
    """Wait for the space to build"""
    print("⏳ Space is building... This may take a few minutes.")
    print(f"🌐 You can monitor the build at: https://huggingface.co/spaces/{SPACE_NAME}")
    print("📱 The space will be available once the build completes.")
    print("\n⚡ Note: First startup may take longer due to model downloads")

def main():
    """Main deployment function"""
    print("🎙️ MeetingNotes - HuggingFace Spaces Deployment")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check dependencies
    check_dependencies()
    
    # Get HuggingFace token
    token = get_hf_token()
    
    # Validate source structure
    validate_source_structure()
    
    # Create space configuration
    create_space_config()
    
    # Validate files
    validate_files()
    
    # Create or update space
    update_mode = create_space(token)
    
    # Deploy files
    deploy_files(token)
    
    # Success message
    print("\n🎉 Deployment completed successfully!")
    print(f"🌐 Space URL: https://huggingface.co/spaces/{SPACE_NAME}")
    
    if not update_mode:
        wait_for_space_build()
    
    print(f"\n📱 Your MeetingNotes app is now live at:")
    print(f"   https://huggingface.co/spaces/{SPACE_NAME}")
    print(f"\n🚀 Space deployed with ZeroGPU for optimal performance!")
    print(f"🔧 Native MCP server enabled for programmatic access!")
    print("\n🎙️ Happy meeting analyzing! ✨")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)