# MeetingNotes - Analyse Intelligente de Réunions avec Voxtral

Application web utilisant l'IA **Voxtral de Mistral AI** pour analyser automatiquement vos réunions audio/vidéo avec :
- **Analyse directe** : Transcription et résumé structuré en une seule étape
- **3 modes de traitement** : Local (Transformers), MLX (Apple Silicon), API (Cloud)
- **Modèles quantifiés** : Support 4bit/8bit pour économie mémoire
- **Diarisation intelligente** : Identification et renommage des locuteurs
- **Résumés personnalisables** : Sections modulaires selon vos besoins

## 🚀 Installation Rapide

1. **Clonez le repository et installez les dépendances :**
```bash
git clone <repository-url>
cd meetingnotes
pip install -r requirements.txt
```

2. **Configurez votre token Hugging Face :**
```bash
cp .env.example .env
# Éditez .env et ajoutez votre token Hugging Face
```

3. **Lancez l'application :**
```bash
python main.py
```

L'interface web sera accessible sur **http://localhost:7860**

## ⚙️ Configuration

### Token Hugging Face (Obligatoire)
Obtenez un token d'accès sur [Hugging Face](https://huggingface.co/settings/tokens) et ajoutez-le dans `.env` :
```env
HUGGINGFACE_TOKEN=votre_token_ici
```

### Clé API Mistral (Optionnelle)
Pour utiliser le mode API cloud, obtenez une clé sur [Mistral AI](https://console.mistral.ai/) :
```env
MISTRAL_API_KEY=votre_cle_api_mistral
```

## 🎯 Fonctionnalités

### Modes de Traitement
- **Local (Transformers)** : Traitement sur votre machine avec PyTorch
- **MLX (Apple Silicon)** : Optimisé pour Mac M1/M2/M3 avec MLX Framework
- **API (Cloud)** : Utilise l'API Mistral Cloud

### Modèles et Quantification
| Modèle | Précision | Repository | Usage Mémoire |
|--------|-----------|------------|---------------|
| **Voxtral Mini** | Default | `mistralai/Voxtral-Mini-3B-2507` | ~6GB |
| **Voxtral Mini** | 8bit | `mzbac/voxtral-mini-3b-8bit` | ~3.5GB |
| **Voxtral Mini** | 4bit | `mzbac/voxtral-mini-3b-4bit-mixed` | ~2GB |
| **Voxtral Small** | Default | `mistralai/Voxtral-Small-24B-2507` | ~48GB |
| **Voxtral Small** | 8bit | `VincentGOURBIN/voxtral-small-8bit` | ~24GB |
| **Voxtral Small** | 4bit | `VincentGOURBIN/voxtral-small-4bit-mixed` | ~12GB |

### Diarisation des Locuteurs
- **Identification automatique** : Détection des différents locuteurs avec pyannote.audio
- **Segments de référence** : Écoute d'échantillons audio pour chaque locuteur
- **Renommage personnalisé** : Attribution de noms humains aux locuteurs
- **Intégration contexte** : Utilisation des informations de locuteurs dans les résumés

### Résumés Personnalisables
**Sections modulaires** : Choisissez les sections à inclure selon vos besoins
- **📄 Résumé exécutif** : Aperçu global de la réunion
- **💬 Discussions principales** : Sujets principaux abordés
- **✅ Plan d'action** : Actions, responsabilités, échéances
- **⚖️ Décisions prises** : Décisions validées
- **⏭️ Prochaines étapes** : Actions de suivi
- **📌 Sujets principaux** : Informations présentées
- **⭐ Points importants** : Insights et données clés
- **❓ Questions & discussions** : Questions posées et réponses
- **📝 Éléments de suivi** : Clarifications nécessaires

**Profils prédéfinis** :
- **🎯 Profil Action** : Focus sur les tâches et décisions
- **📊 Profil Information** : Focus sur les données et insights
- **📋 Profil Complet** : Toutes les sections activées

### Formats Supportés
- **Audio** : WAV, MP3, M4A, OGG, FLAC
- **Vidéo** : MP4, AVI, MOV, MKV (extraction automatique de l'audio)

## 🔧 Utilisation

### 1. Configuration du Mode de Traitement
1. **Choisissez le mode** : Local, MLX ou API
2. **Sélectionnez le modèle** : Mini ou Small selon vos besoins
3. **Choisissez la précision** : Default, 8bit ou 4bit pour optimiser la mémoire

### 2. Upload et Options
- **Fichier** : Audio direct ou vidéo (extraction automatique)
- **Découpe optionnelle** : Trimming début/fin (laissez vide pour 0)
- **Taille chunks** : Durée de traitement (5-25 minutes)

### 3. Diarisation (Optionnel)
1. **Analysez les locuteurs** avec pyannote.audio
2. **Écoutez les segments** de référence de chaque locuteur
3. **Renommez les locuteurs** avec des noms personnalisés
4. **Appliquez les renommages** pour un contexte enrichi

### 4. Personnalisation du Résumé
- **Sections modulaires** : Activez seulement les sections nécessaires
- **Profils rapides** : Action, Information ou Complet
- **Configuration flexible** : Adaptez le résumé à votre usage

### 5. Analyse et Résultats
Cliquez sur **"Analyser la réunion"** pour obtenir un résumé structuré personnalisé.

## 🏗️ Architecture

Le projet suit une architecture modulaire dans `src/meetingnotes/` :

```
src/meetingnotes/
├── ai/                    # Intelligence Artificielle
│   ├── voxtral_analyzer.py      # Analyseur Voxtral local (Transformers)
│   ├── voxtral_api_analyzer.py  # Analyseur Voxtral API
│   ├── voxtral_mlx_analyzer.py  # Analyseur Voxtral MLX (Apple Silicon)
│   ├── diarization.py           # Diarisation des locuteurs (pyannote)
│   ├── memory_manager.py        # Gestion optimisée de la mémoire
│   └── prompts_config.py        # Configuration centralisée des prompts
├── audio/                 # Traitement Audio
│   ├── wav_converter.py         # Conversion de formats
│   └── normalizer.py            # Normalisation du volume
├── core/                  # Logique Métier
│   ├── voxtral_direct.py        # Traitement direct (Transformers)
│   ├── voxtral_api.py           # Interface API Mistral
│   └── voxtral_mlx.py           # Interface MLX Apple Silicon
├── ui/                    # Interface Utilisateur
│   ├── main.py                  # Interface Gradio principale
│   └── handlers.py              # Gestionnaires d'événements
└── utils/                 # Utilitaires
    ├── __init__.py              # Module utils
    └── time_formatter.py        # Formatage des durées
```

Pour plus de détails, consultez [ARCHITECTURE.md](ARCHITECTURE.md).

## 🔧 Configuration Avancée

### Variables d'Environnement
```env
# Obligatoire pour tous les modes
HUGGINGFACE_TOKEN=votre_token_hf

# Optionnel pour le mode API
MISTRAL_API_KEY=votre_cle_mistral
```

### Optimisations selon votre Matériel
- **Mac M1/M2/M3** : Utilisez le mode MLX pour de meilleures performances
- **GPU NVIDIA** : Mode Local avec accélération CUDA automatique
- **CPU seulement** : Privilégiez les modèles 4bit pour économiser la mémoire
- **Mémoire limitée** : Mini 4bit (~2GB) ou Small 4bit (~12GB)

## 🔍 Caractéristiques Techniques

### Optimisations Mémoire
- **Modèles pré-quantifiés** : 4bit et 8bit pour réduction mémoire
- **Gestionnaire de mémoire** : Nettoyage automatique entre les chunks
- **Support multi-plateforme** : MPS (Apple), CUDA (NVIDIA), CPU optimisé

### Traitement Intelligent
- **3 modes d'inférence** : Direct audio-chat sans transcription intermédiaire
- **Chunks adaptatifs** : Division intelligente des longs fichiers avec synthèse
- **Prompts modulaires** : Sections de résumé personnalisables
- **Contexte enrichi** : Intégration de la diarisation dans les analyses

### Interface Moderne
- **Interface intuitive** : Sélection séparée modèle/précision
- **Diarisation interactive** : Écoute et renommage des locuteurs
- **Sections modulaires** : Personnalisation avancée des résumés
- **Feedback temps réel** : Indicateurs de progression détaillés

## 📦 Dépendances Principales

- **gradio** : Interface utilisateur web moderne
- **torch/torchaudio** : Framework de deep learning (mode Local)
- **transformers** : Modèles Hugging Face et Voxtral
- **mlx/mlx-voxtral** : Framework MLX optimisé Apple Silicon (macOS uniquement)
- **pyannote.audio** : Diarisation des locuteurs
- **pydub** : Traitement et conversion audio
- **requests** : Communication avec l'API Mistral
- **python-dotenv** : Gestion des variables d'environnement

## 🔒 Sécurité et Confidentialité

- **Traitement local** : Option de traitement entièrement sur votre machine
- **Variables d'environnement** : Tokens sécurisés via `.env`
- **Pas de stockage cloud** : Vos fichiers restent locaux
- **Nettoyage automatique** : Suppression des fichiers temporaires

## 🚦 Statut du Projet

✅ **Version v2.0** - Fonctionnalités Avancées
- **3 modes de traitement** : Local, MLX, API
- **6 configurations de modèles** : Mini/Small + Default/8bit/4bit  
- **Diarisation complète** : Identification et renommage des locuteurs
- **Résumés modulaires** : 9 sections personnalisables
- **Interface optimisée** : Sélection intuitive modèle/précision
- **Support multi-plateforme** : Windows, macOS, Linux

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créez une branche feature
3. Implémentez vos modifications
4. Ajoutez des tests si nécessaire
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---

**MeetingNotes** - Propulsé par [Voxtral de Mistral AI](https://mistral.ai/) | 🚀 Analyse intelligente de réunions | 💾 Traitement local et cloud sécurisé