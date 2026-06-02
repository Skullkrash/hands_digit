# Description du projet

Ce projet a été réalisé dans le cadre d'un module d'IA en école d'ingérieur. Il consistait à créer un dataset et entraîner un modèle à reconnaître un nombre de doigts montré à la caméra.

## Setup

### API

Les instructions suivantes sont à réaliser de préférence dans un environnement virtuel.

Les dépendances sont inscrites dans le fichier requirements.txt. Pour les installer:
- `pip install -r requirements.txt`

Pour démarrer le serveur la première fois:
- uvicorn app:app --reload --host 0.0.0.0 --port 8000

Pour démarrer le serveur par la suite:
- uvicorn app:app --reload

### Frontend

Prérequis: npm

Dans le dossier frontend:
- `npm run dev`