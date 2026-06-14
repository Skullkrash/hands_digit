# Description du projet

Ce projet a été réalisé dans le cadre d'un module d'IA en école d'ingérieur. Il consistait à créer un dataset et entraîner un modèle à reconnaître un nombre de doigts montré à la caméra.

## Setup

### Projet web

Pour lancer le projet en faisant tourner le modèle sur CPU :

- `docker compose up --build`

Pour lancer le projet en faisant tourner le modèle sur GPU (avec CUDA installé) :

- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`


Le frontend est servi sous localhost:3000, et le backend sous localhost:8000.


## Pipeline de training MLOps

La pipeline est orchestrée via **GitHub Actions** sur un runner auto-hébergé (self-hosted). Elle :

1. Clone le repo sur la machine cible (là où tourne le runner)
2. Crée un environnement virtuel Python et installe les dépendances
3. Lance `pipeline.py` qui déclenche l'entraînement via Ultralytics HUB
4. Upload les artefacts de résultats (runs, rapports)

Le fichier de workflow est `.github/workflows/train_pipeline.yml`.

---

## Prérequis

### Matériel

- GPU NVIDIA compatible CUDA (le job tourne sur un runner taggé `[self-hosted, gpu, cuda]`)
- CUDA 12.6+ installé sur la machine (requis par `torch==2.12.0+cu126`)

### Logiciels

- **Windows** (le workflow utilise PowerShell et des chemins `.venv_train/Scripts/`)
- **Python 3.10+**
- **Git**

### Compte & accès

- Une clé API Ultralytics HUB est définie en tant que secret GitHub sur le projet, permettant son utilisation sans risque dans la pipeline

Le workflow l'injecte directement comme variable d'environnement dans le step `pipeline-run` :

```yaml
- name: pipeline-run
  working-directory: train
  env:
    ULTRALYTICS_API_KEY: ${{ secrets.ULTRALYTICS_API_KEY }}
  run: ./.venv_train/Scripts/python.exe pipeline.py
```

---

## Structure du dossier `train/`

```
train/
├── pipeline.py        # Point d'entrée de la pipeline
├── train.py           # Logique d'entraînement YOLO + téléchargement du dataset
├── config.yaml        # Hyperparamètres et paramètres d'augmentation
└── requirements.txt   # Dépendances Python pour le setup de l'environnement virtuel sur le runner
```

### `config.yaml`

Modifie ce fichier pour ajuster les hyperparamètres sans toucher au code :

---

## Déclencher la pipeline

### Automatiquement

La pipeline se déclenche sur chaque push vers `main` qui modifie des fichiers dans :
- `train/**`
- `.github/workflows/train_pipeline.yml`

### Manuellement

Dans GitHub, va dans **Actions → Hands-digits training pipeline → Run workflow**.

