"""
Utilitaires pour formater les durées en format lisible.

Ce module fournit des fonctions pour convertir les durées en secondes
vers un format lisible en français.
"""


def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes vers un format lisible.
    
    Args:
        seconds (float): Durée en secondes
        
    Returns:
        str: Durée formatée (ex: "7 minutes 48 secondes", "1 minute 5 secondes", "42 secondes")
    """
    if seconds < 0:
        return "0 seconde"
    
    # Convertir en entier pour éviter les décimales
    total_seconds = int(round(seconds))
    
    # Calculer minutes et secondes
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    
    # Construire le format
    parts = []
    
    if minutes > 0:
        if minutes == 1:
            parts.append("1 minute")
        else:
            parts.append(f"{minutes} minutes")
    
    if remaining_seconds > 0:
        if remaining_seconds == 1:
            parts.append("1 seconde")
        else:
            parts.append(f"{remaining_seconds} secondes")
    
    # Cas spécial: 0 seconde
    if not parts:
        return "0 seconde"
    
    # Joindre avec " et " si deux parties
    if len(parts) == 2:
        return f"{parts[0]} et {parts[1]}"
    else:
        return parts[0]


def format_duration_short(seconds: float) -> str:
    """
    Formate une durée en format court (ex: "7m48s", "1m05s", "42s").
    
    Args:
        seconds (float): Durée en secondes
        
    Returns:
        str: Durée formatée en format court
    """
    if seconds < 0:
        return "0s"
    
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    
    if minutes > 0:
        return f"{minutes}m{remaining_seconds:02d}s"
    else:
        return f"{remaining_seconds}s"