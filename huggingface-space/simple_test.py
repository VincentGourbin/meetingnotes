#!/usr/bin/env python3
"""
Simple test to verify the MPS support and interface work correctly.
"""

import gradio as gr
import torch
import os

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    
    # Test device detection
    print("🔍 Device Detection:")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    # Device selection logic (from ZeroGPUManager)
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(f"  ✅ Selected: {device} with {dtype}")
    elif torch.cuda.is_available():
        device = "cuda" 
        dtype = torch.bfloat16
        print(f"  ✅ Selected: {device} with {dtype}")
    else:
        device = "cpu"
        dtype = torch.float16
        print(f"  ⚠️ Selected: {device} with {dtype}")
    
    # Test tensor operations
    try:
        test_tensor = torch.randn(100, 100, dtype=dtype).to(device)
        result = torch.mm(test_tensor, test_tensor)
        print(f"  ✅ Tensor operations work on {device}")
        
        if device == "mps":
            memory = torch.mps.current_allocated_memory() / (1024**2)
            print(f"  📊 MPS Memory: {memory:.1f}MB")
            torch.mps.empty_cache()
    except Exception as e:
        print(f"  ❌ Tensor operations failed: {e}")
    
    # Test HF Spaces detection
    is_spaces = os.getenv("SPACE_ID") is not None
    print(f"  HF Spaces environment: {is_spaces}")
    
    return device, dtype

def create_simple_interface():
    """Create a simple interface to test Gradio."""
    
    def process_info(name, enable_feature, number_input):
        """Simple processing function."""
        if not name:
            return "Please enter a name!"
        
        result = f"Hello {name}!\n"
        result += f"Feature enabled: {enable_feature}\n"
        result += f"Number: {number_input}\n"
        result += f"Device detected: {torch.backends.mps.is_available() and 'MPS' or 'CPU'}"
        
        return result
    
    with gr.Blocks(title="MeetingNotes Test", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧪 MeetingNotes - Simple Test Interface")
        gr.Markdown("Testing MPS support and Gradio interface without heavy AI dependencies.")
        
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(label="Your Name", placeholder="Enter your name")
                enable_checkbox = gr.Checkbox(label="Enable Test Feature", value=True)
                number_input = gr.Number(label="Test Number", value=42, minimum=0, maximum=100)
                
            with gr.Column():
                output_text = gr.Textbox(label="Output", lines=10, interactive=False)
                process_btn = gr.Button("Process", variant="primary")
        
        process_btn.click(
            fn=process_info,
            inputs=[name_input, enable_checkbox, number_input],
            outputs=[output_text]
        )
        
        gr.Markdown("""
        **Status**: This test interface verifies that:
        - MPS device detection works
        - Gradio interface components work
        - Basic torch operations work on MPS
        """)
    
    return interface

def main():
    """Main test function."""
    print("🚀 MeetingNotes HF Spaces - Simple Test")
    print("=" * 50)
    
    # Test basic functionality
    device, dtype = test_basic_functionality()
    
    print("\n🎨 Creating test interface...")
    interface = create_simple_interface()
    
    print("🌐 Launching interface on http://localhost:7862")
    print("   (Press Ctrl+C to stop)")
    
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            show_error=True,
            quiet=False,
            mcp_server=False  # No MCP for simple test
        )
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")

if __name__ == "__main__":
    main()