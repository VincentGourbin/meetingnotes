#!/usr/bin/env python3
"""
Point d'entrée principal de l'application MeetingNotes.

Ce script lance l'interface Gradio pour le traitement de réunions audio/vidéo.
"""

import sys
import os

# Ajouter le dossier src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from meetingnotes.ui.main import main

if __name__ == "__main__":
    main()