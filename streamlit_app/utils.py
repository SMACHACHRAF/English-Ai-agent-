"""
Fonctions utilitaires pour l'application Streamlit.
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from nltk.tokenize import word_tokenize
import re
import streamlit as st

def custom_css():
    """
    Retourne le CSS personnalisé pour l'application.
    """
    return """
    <style>
        .main-title {
            color: #1E88E5;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        
        .chat-message.user {
            background-color: #2b313e;
        }
        
        .chat-message.bot {
            background-color: #475063;
        }
        
        .avatar {
            width: 55px;
            height: 55px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 1rem;
        }
        
        .message {
            flex: 1;
        }
        
        .stButton>button {
            width: 100%;
        }
        
        .level-info {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            background-color: #3b4253;
        }
        
        /* Gamification elements */
        .achievement {
            padding: 0.5rem;
            background-color: #4CAF50;
            border-radius: 0.3rem;
            color: white;
            margin: 0.2rem;
            display: inline-block;
        }
        
        /* Improving readability */
        .stTextInput>div>div>input {
            padding: 0.5rem 1rem;
            font-size: 1.1rem;
        }
        
        /* Make examples look like buttons */
        .example-question {
            background-color: #475063;
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
        }
        
        .example-question:hover {
            background-color: #5b6882;
        }
    </style>
    """

def get_random_examples(qa_data, num_examples=5):
    """
    Récupère un échantillon aléatoire de questions-réponses.
    """
    if len(qa_data) <= num_examples:
        return qa_data
    return random.sample(qa_data, num_examples)

def detect_english_level(text):
    """
    Détecte le niveau d'anglais d'un texte basé sur des heuristiques simples.
    
    Cette fonction est un exemple simplifié. Un modèle plus sophistiqué pourrait être utilisé.
    """
    # Liste de mots/structures par niveau
    beginner_words = set(['hello', 'yes', 'no', 'thank', 'please', 'what', 'where', 'when', 'how', 'who', 'is', 'are', 'am'])
    elementary_words = set(['because', 'but', 'then', 'there', 'here', 'now', 'yesterday', 'tomorrow', 'like', 'love', 'hate'])
    intermediate_words = set(['however', 'therefore', 'although', 'despite', 'nevertheless', 'furthermore', 'moreover'])
    advanced_words = set(['consequently', 'presumably', 'accordingly', 'alternatively', 'subsequently', 'notwithstanding'])
    
    # Nettoyage et tokenisation
    text = text.lower()
    words = set(word_tokenize(text))
    
    # Vérification des mots par niveau
    beginner_count = len(words.intersection(beginner_words))
    elementary_count = len(words.intersection(elementary_words))
    intermediate_count = len(words.intersection(intermediate_words))
    advanced_count = len(words.intersection(advanced_words))
    
    # Longueur moyenne des phrases (indicateur de complexité)
    sentences = re.split(r'[.!?]+', text)
    avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences if s) / len([s for s in sentences if s])
    
    # Complexité du vocabulaire (approximation simplifiée)
    vocab_complexity = len(set(word_tokenize(text))) / len(word_tokenize(text)) if word_tokenize(text) else 0
    
    # Évaluation du niveau
    if advanced_count > 0 and avg_sentence_length > 15 and vocab_complexity > 0.7:
        return "Advanced"
    elif intermediate_count > 0 and avg_sentence_length > 10 and vocab_complexity > 0.6:
        return "Intermediate"
    elif elementary_count > 0 or avg_sentence_length > 7:
        return "Elementary"
    else:
        return "Beginner"

def create_level_progress_chart(user_history):
    """
    Crée un graphique montrant la progression du niveau d'anglais de l'utilisateur.
    """
    if not user_history:
        return None
    
    # Préparation des données
    dates = [entry['date'] for entry in user_history]
    levels = [entry['level'] for entry in user_history]
    
    # Conversion des niveaux en valeurs numériques
    level_values = {
        'Beginner': 1,
        'Elementary': 2,
        'Intermediate': 3,
        'Advanced': 4,
        'Proficient': 5
    }
    
    numerical_levels = [level_values.get(level, 0) for level in levels]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, numerical_levels, marker='o', linestyle='-', color='#1E88E5', linewidth=2, markersize=8)
    ax.set_title('Your English Level Progression', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Level', fontsize=12)
    ax.set_yticks(list(level_values.values()))
    ax.set_yticklabels(list(level_values.keys()))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def create_word_embeddings_visualization(words, embeddings_model):
    """
    Crée une visualisation t-SNE des embeddings de mots pour montrer les relations entre les mots.
    """
    # Obtenir les embeddings pour chaque mot
    word_vectors = []
    valid_words = []
    
    for word in words:
        try:
            vector = embeddings_model[word]
            word_vectors.append(vector)
            valid_words.append(word)
        except KeyError:
            continue
    
    if len(word_vectors) < 2:
        return None
    
    # Réduire la dimensionnalité avec t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(word_vectors)-1, 30))
    reduced_vectors = tsne.fit_transform(np.array(word_vectors))
    
    # Création du graphique de dispersion
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Afficher les points
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)
    
    # Ajouter les étiquettes des mots
    for i, word in enumerate(valid_words):
        ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)
    
    ax.set_title('Word Relationships Visualization', fontsize=16)
    ax.set_xlabel('t-SNE dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE dimension 2', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig

def generate_exercise(topic, level="Intermediate"):
    """
    Génère un exercice simple basé sur un sujet et un niveau.
    """
    # Exemple simplifié - dans une application réelle, ceci pourrait être généré par un modèle
    exercises = {
        "present perfect": {
            "Beginner": {
                "title": "Complete with have/has + past participle",
                "questions": [
                    "I ____ (never/visit) Paris.",
                    "She ____ (just/finish) her homework.",
                    "We ____ (already/see) that movie.",
                ],
                "answers": ["have never visited", "has just finished", "have already seen"]
            },
            "Intermediate": {
                "title": "Present Perfect vs Past Simple",
                "questions": [
                    "I ____ (live) in London since 2015.",
                    "She ____ (visit) Paris last summer.",
                    "They ____ (not see) each other for years.",
                ],
                "answers": ["have lived", "visited", "haven't seen"]
            },
            "Advanced": {
                "title": "Present Perfect Continuous vs Present Perfect Simple",
                "questions": [
                    "She ____ (teach) English for ten years.",
                    "How many books ____ you ____ (read) this month?",
                    "I ____ (wait) for you for two hours!",
                ],
                "answers": ["has been teaching", "have you read", "have been waiting"]
            }
        },
        "vocabulary": {
            "Beginner": {
                "title": "Match the opposites",
                "questions": ["hot", "big", "happy"],
                "options": ["sad", "small", "cold"],
                "answers": ["cold", "small", "sad"]
            },
            "Intermediate": {
                "title": "Fill in with the correct word",
                "questions": [
                    "She's very ____ (reliable/trustworthy); you can always count on her.",
                    "The film was so ____ (boring/dull) that I fell asleep.",
                    "This soup is ____ (delicious/tasty); I love it!",
                ],
                "answers": ["reliable", "boring", "delicious"]
            }
        }
    }
    
    # Recherche d'exercices correspondant au sujet
    for subject, level_exercises in exercises.items():
        if subject in topic.lower() and level in level_exercises:
            return level_exercises[level]
    
    # Exercice par défaut si rien ne correspond
    return {
        "title": f"{topic.capitalize()} Exercise ({level})",
        "questions": [
            "This is an automatically generated exercise.",
            "In a real application, this would be created by the AI model.",
            "Based on your question about: " + topic
        ],
        "answers": ["Answer 1", "Answer 2", "Answer 3"]
    }

def summarize_course(text, level="Intermediate"):
    """
    Crée un résumé d'un cours en fonction du niveau de l'utilisateur.
    """
    # Dans une application réelle, cette fonction utiliserait un modèle pour résumer le contenu
    # Ceci est une version simplifiée pour la démonstration
    
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2:
        return text
    
    # Ajuster la longueur du résumé en fonction du niveau
    if level in ["Beginner", "Elementary"]:
        return "\n\n".join(paragraphs[:2])  # Version très courte pour débutants
    elif level == "Intermediate":
        return "\n\n".join(paragraphs[:4])  # Version moyenne pour niveau intermédiaire
    else:  # Advanced, Proficient
        return "\n\n".join(paragraphs[:6])  # Version plus complète pour niveaux avancés
        
_model_cache = {}

def generate_ai_response(prompt, level="Intermediate"):
    """
    Génère une réponse à partir du modèle LLM adapté.
    Cette fonction utilise le modèle Mistral fine-tuné pour générer des réponses appropriées
    au niveau d'anglais de l'utilisateur.
    
    Args:
        prompt (str): La question ou l'instruction de l'utilisateur
        level (str): Le niveau d'anglais (Beginner, Elementary, Intermediate, Advanced)
        
    Returns:
        str: La réponse générée par le modèle
    """
    import torch
    import os
    import logging
    import sys
    import time
    
    # Configurer le logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("Transformers library not found. Please install it with: pip install transformers")
        return f"I'm having trouble accessing the language model. Here's a helpful response for your {level} level: Please ask your question again or try a different topic."
    
    # Vérifier si le prompt est vide ou trop court
    if not prompt or len(prompt) < 2:
        return f"I need a bit more information. Could you please elaborate on your question? I'm here to help with your {level} level English practice."
    
    try:
        # Chemin vers le modèle fine-tuné (adaptateurs LoRA)
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        if not os.path.exists(model_dir):
            logger.error(f"Model directory not found: {model_dir}")
            return f"I'm currently unable to access my language model. Let me provide a general response for your {level} level: English learning is a journey of practice and discovery. How else can I assist you today?"
        
        # Cache pour le modèle et le tokenizer
        global _model_cache
        if not _model_cache:
            start_time = time.time()
            logger.info(f"Loading model from {model_dir}...")
            
            try:
                # Charger le tokenizer d'abord (plus léger)
                tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                
                # Configurer le modèle pour l'inférence
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Configurer le modèle pour l'inférence
                if device == "cuda":
                    # Version GPU avec précision mixte
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    # Version CPU - ajustements pour économiser la mémoire
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                
                # Préparer pour l'inférence
                model.eval()
                
                # Mettre en cache
                _model_cache = {"model": model, "tokenizer": tokenizer}
                
                logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return f"I'm experiencing technical difficulties with my language model. Here's a helpful response for your {level} level: Could you rephrase your question or try asking about a different topic?"
        else:
            model = _model_cache["model"]
            tokenizer = _model_cache["tokenizer"]
            logger.info("Using cached model")
        
        # Préparer le contexte selon le niveau
        level_context = f"You are an English language tutor for {level} level students. Respond in a way that's appropriate for their level."
        
        # Formatter le prompt pour la génération avec le format d'instruction de Mistral
        formatted_prompt = f"<s>[INST] {prompt}\n\nContext: {level_context} [/INST]"
        
        # Tokeniser l'entrée
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Générer la réponse
        try:
            with torch.no_grad():
                start_time = time.time()
                logger.info("Generating response...")
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                    num_return_sequences=1
                )
                
                logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
            
            # Décoder la réponse générée
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire uniquement la réponse générée (après le prompt)
            # Mistral répète parfois le prompt dans la sortie
            response = generated_text[len(formatted_prompt):].strip()
            
            # Supprimer le texte du prompt s'il apparaît dans la réponse
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Vérifier si la réponse est vide
            if not response:
                logger.warning("Generated empty response")
                return f"I'm thinking about your question. In the meantime, could you provide more details about what you'd like to learn in {level} level English?"
                
            return response
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"I encountered an issue while generating a response. Let me provide a general answer for your {level} level: English learning combines vocabulary, grammar, and practice. What specific aspect would you like to focus on today?"
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"I'm currently experiencing some technical issues. As your {level} level English tutor, I'd be happy to help you once the system is back online. In the meantime, could you tell me what English topics you're interested in?"