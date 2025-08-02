"""
Centralized prompts configuration for Voxtral in HF Spaces.

This module contains all prompts used by Voxtral analyzers
for different types of analyses and processing modes.
"""


class VoxtralPrompts:
    """Class containing all system prompts for Voxtral."""
    
    # ====================================
    # AVAILABLE SECTIONS FOR SUMMARIES
    # Note: Titles are in English but the AI will adapt language based on meeting content
    # ====================================
    
    AVAILABLE_SECTIONS = {
        "resume_executif": {
            "title": "## EXECUTIVE SUMMARY",
            "description": "Overview of the purpose of this meeting segment and its outcomes",
            "default_action": True,
            "default_info": True
        },
        "discussions_principales": {
            "title": "## MAIN DISCUSSIONS", 
            "description": "Main topics addressed and important points raised",
            "default_action": True,
            "default_info": False
        },
        "sujets_principaux": {
            "title": "## MAIN TOPICS",
            "description": "Key topics discussed and information presented", 
            "default_action": False,
            "default_info": True
        },
        "plan_action": {
            "title": "## ACTION PLAN",
            "description": "Complete list of actions with:\n- Specific tasks and deliverables\n- Assigned responsibilities\n- Deadlines and timelines\n- Priority levels",
            "default_action": True,
            "default_info": False
        },
        "decisions_prises": {
            "title": "## DECISIONS MADE",
            "description": "All decisions made during this segment",
            "default_action": True,
            "default_info": False
        },
        "points_importants": {
            "title": "## KEY POINTS", 
            "description": "Important discoveries, data or insights shared",
            "default_action": False,
            "default_info": True
        },
        "questions_discussions": {
            "title": "## QUESTIONS & DISCUSSIONS",
            "description": "Main questions asked and discussions held",
            "default_action": False,
            "default_info": True
        },
        "prochaines_etapes": {
            "title": "## NEXT STEPS",
            "description": "Follow-up actions and planned future meetings", 
            "default_action": True,
            "default_info": False
        },
        "elements_suivi": {
            "title": "## FOLLOW-UP ELEMENTS",
            "description": "Follow-up information or clarifications needed",
            "default_action": False,
            "default_info": True
        }
    }
    
    @staticmethod
    def get_meeting_summary_prompt(selected_sections: list, speaker_references: str = "", chunk_info: str = "", previous_context: str = "") -> str:
        """
        Generate meeting summary prompt according to selected sections.
        
        Args:
            selected_sections (list): List of section keys to include
            speaker_references (str): Diarization context with tags (optional)
            chunk_info (str): Audio segment information (optional)
            previous_context (str): Context from previous segments (optional)
            
        Returns:
            str: Formatted prompt
        """
        # Diarization context
        diarization_context = ""
        if speaker_references and speaker_references.strip():
            diarization_context = f"""

CONTEXT FOR YOUR ANALYSIS (do not include in your response):
Different speakers have been automatically identified in the audio: {speaker_references}
Use this information to enrich your analysis but do not display it in your final response.

"""
        
        # Previous segments context
        previous_summary_context = ""
        if previous_context and previous_context.strip():
            previous_summary_context = f"""

CONTEXT FROM PREVIOUS SEGMENTS (do not include in your response):
Here's what happened in previous audio segments:
{previous_context}

Use this information to ensure continuity and avoid repetitions, but focus on the new content of this segment.

"""
        
        # Audio segment information
        segment_context = ""
        if chunk_info and chunk_info.strip():
            segment_context = f"""

IMPORTANT: You are analyzing a segment ({chunk_info}) extracted from a longer audio recording.
This segment may start or end in the middle of sentences/discussions.
Focus on the content of this segment while keeping in mind it's part of a larger whole.

"""
        
        # Build selected sections
        sections_text = ""
        for section_key in selected_sections:
            if section_key in VoxtralPrompts.AVAILABLE_SECTIONS:
                section = VoxtralPrompts.AVAILABLE_SECTIONS[section_key]
                sections_text += f"\n{section['title']}\n{section['description']}\n"
                print(f"✅ Section added: {section['title']}")
            else:
                print(f"❌ Unknown section: {section_key}")
        
        return f"""Écoute attentivement ce segment audio de réunion et fournis un résumé structuré complet.{diarization_context}{previous_summary_context}{segment_context}

INSTRUCTION CRITIQUE - LANGUE DE RÉPONSE :
- DÉTECTE la langue parlée dans cet audio
- RÉPONDS OBLIGATOIREMENT dans cette même langue détectée
- Si l'audio est en français → réponds en français
- Si l'audio est en anglais → réponds en anglais  
- Si l'audio est dans une autre langue → réponds dans cette langue
- N'utilise JAMAIS une autre langue que celle détectée dans l'audio

{sections_text}
Formate ta réponse en markdown exactement comme montré ci-dessus."""
    
    @staticmethod
    def get_default_sections(meeting_type: str) -> list:
        """
        Return default sections according to meeting type.
        
        Args:
            meeting_type (str): "action" or "information"
            
        Returns:
            list: List of default section keys
        """
        if "action" in meeting_type.lower():
            return [key for key, section in VoxtralPrompts.AVAILABLE_SECTIONS.items() 
                   if section["default_action"]]
        else:
            return [key for key, section in VoxtralPrompts.AVAILABLE_SECTIONS.items() 
                   if section["default_info"]]
    
    @staticmethod
    def get_synthesis_prompt(selected_sections: list, chunk_summaries: list) -> str:
        """
        Generate prompt for synthesizing multiple chunk summaries.
        
        Args:
            selected_sections (list): List of requested section keys
            chunk_summaries (list): List of chunk summaries to synthesize
            
        Returns:
            str: Formatted synthesis prompt
        """
        # Build selected sections
        sections_text = ""
        for section_key in selected_sections:
            if section_key in VoxtralPrompts.AVAILABLE_SECTIONS:
                section = VoxtralPrompts.AVAILABLE_SECTIONS[section_key]
                sections_text += f"\n{section['title']}\n{section['description']}\n"
        
        # Assembler tous les résumés de chunks
        all_chunks_text = "\n\n=== SÉPARATEUR DE SEGMENT ===\n\n".join(chunk_summaries)
        
        return f"""Tu vas recevoir plusieurs analyses de segments d'une même réunion audio.
Ton rôle est de les synthétiser en un résumé global cohérent et structuré.

ANALYSES DES SEGMENTS À SYNTHÉTISER :
{all_chunks_text}

INSTRUCTION CRITIQUE - LANGUE DE RÉPONSE :
- DÉTECTE la langue utilisée dans les segments ci-dessus
- RÉPONDS OBLIGATOIREMENT dans cette même langue détectée
- Si les segments sont en français → réponds en français
- Si les segments sont en anglais → réponds en anglais
- Évite les répétitions entre segments
- Identifie les éléments récurrents et unifie-les
- Assure la cohérence temporelle et logique
- Produis un résumé global qui reflète l'ensemble de la réunion

Génère un résumé final structuré selon ces sections :
{sections_text}
Formate ta réponse en markdown exactement comme montré ci-dessus."""