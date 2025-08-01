"""
Configuration centralisée des prompts pour Voxtral.

Ce module contient tous les prompts utilisés par les analyseurs Voxtral
pour différents types d'analyses et modes de traitement.
"""


class VoxtralPrompts:
    """Classe contenant tous les prompts système pour Voxtral."""
    
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
        Génère le prompt de résumé de réunion selon les sections sélectionnées.
        
        Args:
            selected_sections (list): Liste des clés de sections à inclure
            speaker_references (str): Contexte de diarisation avec balises (optionnel)
            chunk_info (str): Information sur le segment audio (optionnel)
            previous_context (str): Contexte des segments précédents (optionnel)
            
        Returns:
            str: Prompt formaté
        """
        # Contexte de diarisation
        diarization_context = ""
        if speaker_references and speaker_references.strip():
            diarization_context = f"""

CONTEXTE POUR TON ANALYSE (ne pas inclure dans ta réponse) :
Les différents locuteurs ont été identifiés automatiquement dans l'audio : {speaker_references}
Utilise ces informations pour enrichir ton analyse mais ne les affiche pas dans ta réponse finale.

"""
        
        # Contexte des segments précédents
        previous_summary_context = ""
        if previous_context and previous_context.strip():
            previous_summary_context = f"""

CONTEXTE DES SEGMENTS PRÉCÉDENTS (ne pas inclure dans ta réponse) :
Voici ce qui s'est passé dans les segments audio précédents :
{previous_context}

Utilise ces informations pour assurer la continuité et éviter les répétitions, mais concentre-toi sur le nouveau contenu de ce segment.

"""
        
        # Information sur le segment audio
        segment_context = ""
        if chunk_info and chunk_info.strip():
            segment_context = f"""

IMPORTANT : Tu analyses un segment ({chunk_info}) extrait d'un enregistrement audio plus long.
Ce segment peut commencer ou se terminer au milieu de phrases/discussions.
Concentre-toi sur le contenu de ce segment tout en gardant à l'esprit qu'il fait partie d'un ensemble plus large.

"""
        
        # Construction des sections sélectionnées
        sections_text = ""
        for section_key in selected_sections:
            if section_key in VoxtralPrompts.AVAILABLE_SECTIONS:
                section = VoxtralPrompts.AVAILABLE_SECTIONS[section_key]
                sections_text += f"\n{section['title']}\n{section['description']}\n"
                print(f"✅ Section ajoutée: {section['title']}")
            else:
                print(f"❌ Section inconnue: {section_key}")
        
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
        Retourne les sections par défaut selon le type de réunion.
        
        Args:
            meeting_type (str): "action" ou "information"
            
        Returns:
            list: Liste des clés de sections par défaut
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
        Génère le prompt pour synthétiser plusieurs résumés de chunks.
        
        Args:
            selected_sections (list): Liste des clés de sections demandées
            chunk_summaries (list): Liste des résumés de chunks à synthétiser
            
        Returns:
            str: Prompt de synthèse formaté
        """
        # Construction des sections sélectionnées
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
