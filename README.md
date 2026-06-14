# Description du projet

Ce projet a été réalisé dans le cadre d'un module d'IA en école d'ingérieur. Il consistait à créer un dataset et entraîner un modèle à reconnaître un nombre de doigts montré à la caméra.

## Setup

### Projet web

Pour lancer le projet en faisant tourner le modèle sur CPU :

- `docker compose up --build`

Pour lancer le projet en faisant tourner le modèle sur GPU (avec CUDA installé) :

- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`


Le frontend est servi sous localhost:3000, et le backend sous localhost:8000.


### Pipeline de training MLOps



