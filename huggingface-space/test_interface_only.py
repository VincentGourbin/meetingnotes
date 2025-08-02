#!/usr/bin/env python3
"""
Test interface creation without dependencies that might cause issues.
"""

import gradio as gr

def create_test_interface():
    """Create a minimal test interface."""
    
    # Available analysis sections
    section_choices = [
        ("Executive Summary", "resume_executif"),
        ("Main Discussions", "discussions_principales"),
        ("Key Topics", "sujets_principaux"),
        ("Action Plan", "plan_action"),
        ("Decisions Made", "decisions_prises"),
        ("Important Points", "points_importants"),
        ("Questions & Discussions", "questions_discussions"),
        ("Next Steps", "prochaines_etapes"),
        ("Follow-up Elements", "elements_suivi")
    ]
    
    # Model choices (8-bit only)
    model_choices = [
        ("Voxtral Mini 3B (Fast)", "Voxtral-Mini-3B-2507"),
        ("Voxtral Small 24B (Better Quality)", "Voxtral-Small-24B-2507")
    ]
    
    def dummy_analyze(audio_file, selected_sections, model_name, enable_diarization, num_speakers):
        """Dummy analysis function for testing."""
        if audio_file is None:
            return "", "❌ Please upload an audio file first."
        
        return f"""# Test Analysis Result

**Model Used**: {model_name}
**Sections**: {', '.join(selected_sections)}
**Diarization**: {'Enabled' if enable_diarization else 'Disabled'}
**Speakers**: {num_speakers if num_speakers else 'Auto-detect'}

## Executive Summary
This is a test analysis result. The actual analysis would be performed by Voxtral AI.

## Main Discussions
- Test point 1
- Test point 2
- Test point 3

## Action Plan
- [ ] Test action 1
- [ ] Test action 2
""", "✅ Test analysis completed successfully!"
    
    def clear_results():
        """Clear all results and reset interface."""
        return "", "🗑️ Results cleared"
    
    # Create Gradio interface
    with gr.Blocks(
        title="🎤 MeetingNotes - Intelligent Meeting Analysis",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown("# 🎤 MeetingNotes - Intelligent Meeting Analysis")
        gr.Markdown("""
        Transform your meeting recordings into structured summaries with AI-powered analysis.
        Upload an audio file and get comprehensive insights from your meetings.
        
        **Powered by Voxtral AI** | **Optimized for Hugging Face Spaces** | **Test Version**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Audio input
                gr.Markdown("### 📁 Audio Input")
                gr.Markdown("*Supported formats: MP3, WAV, M4A, OGG (max 25 minutes)*")
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                
                # Model selection
                gr.Markdown("### 🤖 Model Selection") 
                gr.Markdown("*8-bit quantized models for memory efficiency*")
                model_selector = gr.Radio(
                    choices=model_choices,
                    value="Voxtral-Mini-3B-2507",
                    label="Analysis Model"
                )
                
                # Speaker diarization options
                with gr.Group():
                    gr.Markdown("### 🎤 Speaker Options")
                    enable_diarization = gr.Checkbox(
                        label="🎤 Enable Speaker Identification",
                        value=False
                    )
                    num_speakers = gr.Number(
                        label="👥 Number of Speakers (optional)",
                        value=None,
                        precision=0,
                        minimum=2,
                        maximum=10,
                        visible=False
                    )
                
                # Show/hide speaker count based on diarization toggle
                enable_diarization.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enable_diarization],
                    outputs=[num_speakers]
                )
                
            with gr.Column(scale=1):
                # Analysis sections
                gr.Markdown("### 📋 Analysis Sections")
                gr.Markdown("*Select which sections to include in the meeting analysis*")
                sections_selector = gr.CheckboxGroup(
                    choices=section_choices,
                    value=["resume_executif", "discussions_principales", "plan_action"],
                    label="Selected Sections"
                )
                
                # Control buttons
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🚀 Analyze Meeting (Test)",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear Results",
                        variant="secondary"
                    )
        
        # Results section
        with gr.Group():
            gr.Markdown("### 📊 Analysis Results")
            
            status_display = gr.Textbox(
                label="Status",
                value="Ready to analyze (test mode)",
                interactive=False
            )
            
            results_display = gr.Markdown(
                label="Analysis Results",
                value="Upload an audio file and click 'Analyze Meeting' to get started. This is a test interface."
            )
        
        # Event handlers
        analyze_btn.click(
            fn=dummy_analyze,
            inputs=[
                audio_input,
                sections_selector,
                model_selector,
                enable_diarization,
                num_speakers
            ],
            outputs=[results_display, status_display],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_results,
            outputs=[results_display, status_display]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **About**: This is a test interface for MeetingNotes HF Spaces version. 
        The actual application uses Voxtral AI models for intelligent meeting analysis.
        
        **Note**: This test version demonstrates the interface without loading heavy AI models.
        """)
    
    return interface

def main():
    """Main function to test the interface."""
    print("🧪 Testing MeetingNotes Interface (without AI dependencies)")
    
    interface = create_test_interface()
    
    # Launch without MCP server for basic testing
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port for testing
        share=False,
        show_error=True,
        quiet=False,
        mcp_server=False  # Disable MCP for basic test
    )

if __name__ == "__main__":
    main()