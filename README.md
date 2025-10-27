# Projet Chatbot IA pour l'Apprentissage de l'Anglais

Ce projet vise à créer un agent conversationnel (chatbot) pour aider les apprenants à pratiquer l'anglais, basé sur des manuels scolaires de 4 niveaux différents.

## Structure du Projet

- `/data` : Contient les données brutes et prétraitées
  - `data.csv` : Données brutes des manuels
  - `Question_Reponse_DATA_converted.json` : Dataset de questions-réponses au format Alpaca

- `/notebooks` : Notebooks Jupyter pour le prétraitement des données
  - `livrable-finale.ipynb` : Nettoyage de texte et embeddings

- `/training` : Scripts et notebooks pour l'entraînement du modèle LLM
  - `prepare_data.py` : Préparation des données pour fine-tuning
  - `train_model.py` : Script d'entraînement
  - `train_model.ipynb` : Notebook pour entraîner sur Google Colab

- `/models` : Contient les modèles entraînés et les checkpoints

- `/streamlit_app` : Application Streamlit pour l'interface utilisateur
  - `app.py` : Application principale
  - `utils.py` : Fonctions utilitaires
  - `pages/` : Pages additionnelles de l'application

## Installation

```bash
# Installation des dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement du modèle
```bash
cd training
python train_model.py
```

### Démarrer l'application Streamlit
```bash
cd streamlit_app
streamlit run app.py
```

## Fonctionnalités

- Réponse aux questions sur les cours d'anglais
- Adaptation au niveau de l'élève
- Exercices et jeux interactifs
- Interface utilisateur intuitive