# MeetingNotes - Analyse Intelligente de Réunions avec Voxtral

Application web utilisant l'IA **Voxtral de Mistral AI** pour analyser automatiquement vos réunions audio/vidéo avec :
- **Analyse directe** : Transcription et résumé structuré en une seule étape
- **Résumés intelligents** : Comptes-rendus adaptés au type de réunion (action/information)
- **Mode local et API** : Traitement sur votre machine ou via l'API Mistral
- **Affichage des temps lisible** : Durées affichées en format français

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

### Mode de Traitement
- **Local** : Utilise Voxtral-Mini-3B-2507 ou Voxtral-Small-24B-2507 sur votre machine
- **API** : Utilise l'API Mistral Cloud (voxtral-mini-latest ou voxtral-small-latest)

### Types d'Analyse
1. **Analyse Directe** : Transcription + résumé structuré intelligent
2. **Résumés Adaptés** :
   - **Réunion d'information** : Focus sur les insights et discussions
   - **Réunion avec plan d'action** : Focus sur les tâches, décisions et responsabilités

### Formats Supportés
- **Audio** : WAV, MP3, M4A, OGG, FLAC
- **Vidéo** : MP4, AVI, MOV, MKV (extraction automatique de l'audio)

## 🔧 Utilisation

### 1. Mode de Traitement
Choisissez entre **Local** (sur votre machine) ou **API** (cloud Mistral).

### 2. Upload de Fichier
- **Audio** : Déposez directement votre fichier
- **Vidéo** : L'audio sera extrait automatiquement

### 3. Options Avancées
- **Découpe** : Supprimez des secondes au début/fin
- **Taille des chunks** : Ajustez la durée de traitement (5-25 minutes)

### 4. Analyse
Cliquez sur **"Analyser la réunion"** pour obtenir un résumé structuré complet.

## 🏗️ Architecture

Le projet suit une architecture modulaire dans `src/meetingnotes/` :

```
src/meetingnotes/
├── ai/                    # Intelligence Artificielle
│   ├── voxtral_analyzer.py      # Analyseur Voxtral local
│   ├── voxtral_api_analyzer.py  # Analyseur Voxtral API
│   ├── memory_manager.py        # Gestion optimisée de la mémoire
│   └── prompts_config.py        # Configuration centralisée des prompts
├── audio/                 # Traitement Audio
│   ├── wav_converter.py         # Conversion de formats
│   └── normalizer.py            # Normalisation du volume
├── core/                  # Logique Métier
│   ├── voxtral_direct.py        # Traitement direct
│   └── voxtral_api.py           # Interface API Mistral
├── ui/                    # Interface Utilisateur
│   ├── main.py                  # Interface Gradio principale
│   └── handlers.py              # Gestionnaires d'événements
└── utils/                 # Utilitaires
    ├── __init__.py              # Module utils
    └── time_formatter.py        # Formatage des durées
```

Pour plus de détails, consultez [ARCHITECTURE.md](ARCHITECTURE.md).

## 🎛️ Modèles Voxtral

### Mode Local
- **Voxtral-Mini-3B-2507** : Plus rapide, moins de mémoire requise
- **Voxtral-Small-24B-2507** : Plus précis, plus de mémoire requise

### Mode API
- **voxtral-mini-latest** : Plus rapide, moins cher
- **voxtral-small-latest** : Plus précis, plus cher

## 🔍 Caractéristiques Techniques

### Optimisations Mémoire
- **Quantisation INT8** : Réduction de l'empreinte mémoire
- **Gestionnaire de mémoire** : Nettoyage automatique entre les chunks
- **Support MPS/CUDA** : Accélération GPU sur Apple Silicon et NVIDIA

### Traitement Intelligent
- **Chunks adaptatifs** : Division intelligente des longs fichiers
- **Prompts centralisés** : Configuration unifiée pour tous les types d'analyse
- **Formatage temps** : Affichage des durées en format lisible français

### Interface Moderne
- **Interface web responsive** : Design moderne avec Gradio
- **Feedback temps réel** : Indicateurs de progression avec durées lisibles
- **Interface simplifiée** : Focus sur l'analyse directe

## 📦 Dépendances Principales

- **gradio** : Interface utilisateur web moderne
- **torch/torchaudio** : Framework de deep learning
- **transformers** : Modèles Hugging Face et Voxtral
- **pydub** : Traitement et conversion audio
- **requests** : Communication avec l'API Mistral
- **python-dotenv** : Gestion des variables d'environnement

## 🔒 Sécurité et Confidentialité

- **Traitement local** : Option de traitement entièrement sur votre machine
- **Variables d'environnement** : Tokens sécurisés via `.env`
- **Pas de stockage cloud** : Vos fichiers restent locaux
- **Nettoyage automatique** : Suppression des fichiers temporaires

## 🚦 Statut du Projet

✅ **Version Stable** - Prêt pour production
- Architecture modulaire complète
- Support Voxtral local et API
- Interface utilisateur intuitive
- Gestion optimisée de la mémoire
- Documentation complète

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