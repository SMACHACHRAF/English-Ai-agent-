"""
Application Streamlit pour le chatbot d'apprentissage de l'anglais.
"""

import streamlit as st
import os
import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd
import utils

# Configuration de la page
st.set_page_config(
    page_title="English Learning Assistant",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Appliquer le CSS personnalisé
st.markdown(utils.custom_css(), unsafe_allow_html=True)

# Titre de l'application
st.title("🇬🇧 English Learning Assistant")
st.markdown("Your AI-powered English tutor to help you learn and practice English.")

# Chargement du modèle (seulement une fois)
@st.cache_resource
def load_model():
    try:
        # Chemin du modèle
        model_path = "../models"  # Chemin du modèle fine-tuné
        
        # Vérifier si le modèle existe localement
        if not os.path.exists(model_path):
            st.error("Model not found. Please train the model first or update the model path.")
            # Utiliser un modèle de repli si le modèle principal n'est pas disponible
            return None, None
        
        # Charger le modèle BERT pour question-réponse
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Fonction pour générer une réponse
def generate_response(prompt, model, tokenizer, level=None):
    try:
        if model is None or tokenizer is None:
            # Mode de démonstration si le modèle n'est pas disponible
            import random
            import time
            
            demo_responses = {
                "How do I introduce myself in English?": 
                    "To introduce yourself in English, you can say: 'Hello, my name is [your name]. It's nice to meet you.' You can also add information about where you're from or what you do.",
                
                "What's the difference between 'been' and 'gone'?":
                    "'Been' and 'gone' are both past participles but are used differently. 'Been' means someone went somewhere and came back: 'I've been to Paris twice.' 'Gone' means someone went and hasn't returned yet: 'She's gone to the store.'",
                
                "Can you explain the present perfect tense?":
                    "The present perfect tense is formed with 'have/has' + past participle. It's used to talk about: 1) Past experiences (I've visited London), 2) Actions that started in the past and continue now (I've lived here for 5 years), 3) Recent completed actions (I've just finished my homework)."
            }
            
            # Simulation de réflexion
            progress_text = "Thinking..."
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            progress_bar.empty()
            
            # Retourner une réponse de démonstration ou un message générique
            for key, response in demo_responses.items():
                if key.lower() in prompt.lower():
                    return response
            
            return "I can help you with English learning! Please ask me questions about grammar, vocabulary, or language practice."
        
        # Créer un contexte pour la question depuis la base de connaissances
        # Dans une application réelle, on récupérerait le contexte pertinent depuis la base de connaissances
        # Pour cette démonstration, on utilise un contexte générique adapté au niveau
        if level in ["Beginner", "Elementary"]:
            context = "English learning covers basic grammar, vocabulary, and phrases. Beginners focus on simple tenses, common words, and basic conversation. Elementary students learn more words, simple past tense, and can have basic conversations about everyday topics."
        else:
            context = "English learning encompasses grammar rules, vocabulary acquisition, idioms, and language fluency. Advanced learners focus on complex tenses, academic vocabulary, and nuanced expressions. The language has many exceptions to rules and regional variations that affect both written and spoken communication."
        
        # Créer un pipeline de question-réponse
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
        )
        
        # Générer la réponse
        result = qa_pipeline(
            question=prompt,
            context=context,
        )
        
        # Adapter la réponse selon le niveau si spécifié
        response = result["answer"]
        
        # Si le score est trop bas, utiliser une réponse générique
        if result["score"] < 0.1:
            response = "I'm not entirely sure about that. Could you rephrase your question or provide more context?"
            
        return response
    
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while generating a response."

# Charger les exemples de questions-réponses
@st.cache_data
def load_qa_examples():
    try:
        with open("../data/Question_Reponse_DATA_converted.json", 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        return qa_data
    except Exception as e:
        st.warning(f"Could not load QA examples: {e}")
        return []

# Interface principale
def main():
    # Charger le modèle
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    
    # Initialiser l'historique de conversation s'il n'existe pas déjà
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Afficher les messages précédents
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Sélection du niveau dans la sidebar
    with st.sidebar:
        st.subheader("Proficiency Level")
        level = st.select_slider(
            "Select your English level:",
            options=["Beginner", "Elementary", "Intermediate", "Advanced", "Proficient"]
        )
        
        # Ajouter des explications sur les niveaux
        level_descriptions = {
            "Beginner": "Just starting with English basics",
            "Elementary": "Can handle simple exchanges and has basic vocabulary",
            "Intermediate": "Can discuss familiar topics with reasonable fluency",
            "Advanced": "Can communicate effectively with good grammar",
            "Proficient": "Near-native fluency with advanced vocabulary"
        }
        st.info(level_descriptions[level])
        
        # Exemples de questions
        st.subheader("Question Examples")
        qa_examples = load_qa_examples()
        if qa_examples:
            random_examples = utils.get_random_examples(qa_examples, 5)
            for example in random_examples:
                if st.button(f"📝 {example['instruction']}", key=example['instruction']):
                    # Ajouter l'exemple comme entrée de l'utilisateur
                    st.session_state.messages.append({"role": "user", "content": example['instruction']})
                    st.experimental_rerun()
    
    # Zone de saisie de l'utilisateur
    if prompt := st.chat_input("Type your question here..."):
        # Ajouter le message de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Afficher le message de l'utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, model, tokenizer, level)
                st.markdown(response)
        
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Bouton pour approfondir la réponse
        with st.expander("Would you like to know more?"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Examples"):
                    follow_up = f"Could you provide more examples about '{prompt}'?"
                    st.session_state.messages.append({"role": "user", "content": follow_up})
                    st.experimental_rerun()
            with col2:
                if st.button("Explain Simpler"):
                    follow_up = f"Could you explain '{prompt}' in simpler terms?"
                    st.session_state.messages.append({"role": "user", "content": follow_up})
                    st.experimental_rerun()
            with col3:
                if st.button("Practice Exercise"):
                    follow_up = f"Create a practice exercise about '{prompt}'"
                    st.session_state.messages.append({"role": "user", "content": follow_up})
                    st.experimental_rerun()

if __name__ == "__main__":
    main()