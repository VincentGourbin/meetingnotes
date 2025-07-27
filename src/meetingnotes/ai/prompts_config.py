"""
Configuration centralisée des prompts pour Voxtral.

Ce module contient tous les prompts utilisés par les analyseurs Voxtral
pour différents types d'analyses et modes de traitement.
"""


class VoxtralPrompts:
    """Classe contenant tous les prompts système pour Voxtral."""
    
    # ====================================
    # PROMPTS DE RÉSUMÉ DE RÉUNION
    # ====================================
    
    @staticmethod
    def get_meeting_summary_prompt(meeting_type: str, speaker_references: str = "") -> str:
        """
        Génère le prompt de résumé de réunion selon le type.
        
        Args:
            meeting_type (str): "action" ou "information"
            speaker_references (str): Inutilisé (conservé pour compatibilité)
            
        Returns:
            str: Prompt formaté
        """
        if "action" in meeting_type.lower():
            return VoxtralPrompts._get_action_meeting_prompt("")
        else:
            return VoxtralPrompts._get_info_meeting_prompt("")
    
    @staticmethod
    def _get_action_meeting_prompt(speaker_references: str) -> str:
        """Prompt pour réunions axées sur la planification d'actions."""
        return """Écoute ce segment audio de réunion et fournis un résumé structuré complet axé sur la planification d'actions :

## RÉSUMÉ EXÉCUTIF
Aperçu du but de ce segment de réunion et des résultats

## DISCUSSIONS PRINCIPALES
Sujets principaux abordés et points importants soulevés

## PLAN D'ACTION
Liste complète des actions avec :
- Tâches spécifiques et livrables
- Responsabilités assignées
- Échéances et délais
- Niveaux de priorité

## DÉCISIONS PRISES
Toutes les décisions prises pendant ce segment

## PROCHAINES ÉTAPES
Actions de suivi et réunions futures planifiées

Formate ta réponse en markdown exactement comme montré ci-dessus."""
    
    @staticmethod
    def _get_info_meeting_prompt(speaker_references: str) -> str:
        """Prompt pour réunions de partage d'informations."""
        return """Écoute ce segment audio de réunion et fournis un résumé structuré complet pour le partage d'informations :

## RÉSUMÉ EXÉCUTIF
Aperçu du but de ce segment de réunion et des informations clés partagées

## SUJETS PRINCIPAUX
Sujets clés discutés et informations présentées

## POINTS IMPORTANTS
Découvertes importantes, données ou insights partagés

## QUESTIONS & DISCUSSIONS
Questions principales posées et discussions tenues

## ÉLÉMENTS DE SUIVI
Informations de suivi ou clarifications nécessaires

Formate ta réponse en markdown exactement comme montré ci-dessus."""
