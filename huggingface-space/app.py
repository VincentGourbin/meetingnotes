#!/usr/bin/env python3
"""
MeetingNotes - Hugging Face Spaces Version
Intelligent meeting analysis with Voxtral AI, optimized for Zero GPU with native MCP support.
"""

import gradio as gr
import os
from pathlib import Path

# Import custom modules
from src.ui.spaces_interface import create_spaces_interface

def main():
    """Main entry point for the Hugging Face Spaces app with native MCP server."""
    
    # Create the Gradio interface with native MCP support
    interface = create_spaces_interface()
    
    # Launch with specific settings for HF Spaces and MCP server enabled
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False,
        mcp_server=True  # Enable native MCP server
    )

if __name__ == "__main__":
    main()