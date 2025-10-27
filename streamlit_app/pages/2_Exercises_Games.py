"""
Page pour les exercices et jeux interactifs pour l'application Streamlit.
"""

import streamlit as st
import pandas as pd
import os
import json
import random
import time
import sys

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# Configuration de la page
st.set_page_config(
    page_title="Exercises & Games",
    page_icon="🎮",
    layout="wide",
)

# Appliquer le CSS personnalisé
st.markdown(utils.custom_css(), unsafe_allow_html=True)

# Titre de la page
st.title("🎮 Exercises & Interactive Games")
st.markdown("Practice your English with fun exercises and games!")

# Fonction pour charger les données de questions-réponses
@st.cache_data
def load_qa_data():
    try:
        with open("../data/Question_Reponse_DATA_converted.json", 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        return qa_data
    except Exception as e:
        st.warning(f"Could not load QA data: {e}")
        return []

# Charger les données
qa_data = load_qa_data()

# Sidebar pour les options
with st.sidebar:
    st.subheader("Options")
    
    # Sélection du type d'exercice
    exercise_type = st.selectbox(
        "Exercise Type",
        ["Multiple Choice", "Fill in the Blanks", "Matching Pairs", "Word Order", "Flashcards"]
    )
    
    # Sélection du niveau
    level = st.select_slider(
        "Difficulty Level",
        options=["Beginner", "Elementary", "Intermediate", "Advanced", "Proficient"]
    )
    
    # Sélection du sujet/thème
    topics = ["Grammar", "Vocabulary", "Reading", "Writing", "Listening", "Speaking", "Mixed"]
    topic = st.selectbox("Topic", topics)
    
    # Nombre de questions
    num_questions = st.slider("Number of Questions", min_value=5, max_value=20, value=10, step=5)

# Fonction pour générer des exercices à choix multiple
def generate_multiple_choice(data, num_questions, level):
    if not data:
        return []
    
    # Filtrer par niveau (simulation - dans une application réelle, les questions seraient taguées par niveau)
    questions = []
    
    # Sélectionner des questions aléatoires
    selected_items = random.sample(data, min(len(data), num_questions * 3))
    
    for item in selected_items[:num_questions]:
        # Créer des alternatives (incorrect answers) en choisissant d'autres réponses aléatoires
        alternatives = []
        other_outputs = [d["output"] for d in data if d["output"] != item["output"]]
        
        if other_outputs:
            # Sélectionner 3 réponses alternatives aléatoires
            alt_count = min(3, len(other_outputs))
            alternatives = random.sample(other_outputs, alt_count)
        
        # Si nous n'avons pas assez d'alternatives, ajouter des options génériques
        while len(alternatives) < 3:
            alternatives.append(f"None of the above {len(alternatives) + 1}")
        
        # Créer une question
        question = {
            "question": item["instruction"],
            "context": item["input"],
            "correct_answer": item["output"],
            "alternatives": alternatives,
            # Mélanger toutes les options
            "options": [item["output"]] + alternatives
        }
        
        # Mélanger les options pour que la bonne réponse ne soit pas toujours la première
        random.shuffle(question["options"])
        
        questions.append(question)
    
    return questions

# Fonction pour générer des exercices à trous
def generate_fill_in_blanks(data, num_questions, level):
    if not data:
        return []
    
    questions = []
    selected_items = random.sample(data, min(len(data), num_questions * 2))
    
    for item in selected_items[:num_questions]:
        # Prendre la réponse et créer une phrase à trous
        answer = item["output"]
        words = answer.split()
        
        if len(words) < 3:
            continue
        
        # Sélectionner un ou plusieurs mots à remplacer par des blancs
        num_blanks = min(2, len(words) // 2)
        blank_indices = sorted(random.sample(range(len(words)), num_blanks))
        
        # Construire la phrase à trous
        blanked_words = words.copy()
        blank_words = []
        
        for idx in blank_indices:
            blank_words.append(blanked_words[idx])
            blanked_words[idx] = "______"
        
        blanked_text = " ".join(blanked_words)
        
        # Créer une question
        question = {
            "question": item["instruction"],
            "context": item["input"],
            "sentence": blanked_text,
            "missing_words": blank_words
        }
        
        questions.append(question)
    
    return questions

# Fonction pour générer des exercices d'association
def generate_matching_pairs(data, num_questions, level):
    if not data:
        return {}
    
    # Sélectionner des éléments aléatoires
    selected_items = random.sample(data, min(len(data), num_questions))
    
    # Créer les paires
    instructions = [item["instruction"] for item in selected_items]
    outputs = [item["output"] for item in selected_items]
    
    # Mélanger les réponses
    shuffled_outputs = outputs.copy()
    random.shuffle(shuffled_outputs)
    
    return {
        "instructions": instructions,
        "outputs": outputs,
        "shuffled_outputs": shuffled_outputs
    }

# Fonction pour générer des flashcards
def generate_flashcards(data, num_questions, level):
    if not data:
        return []
    
    # Sélectionner des éléments aléatoires
    selected_items = random.sample(data, min(len(data), num_questions))
    
    # Créer les flashcards
    flashcards = []
    for item in selected_items:
        flashcard = {
            "front": item["instruction"],
            "back": item["output"],
            "context": item["input"]
        }
        flashcards.append(flashcard)
    
    return flashcards

# Fonction pour générer des exercices de mise en ordre des mots
def generate_word_order(data, num_questions, level):
    if not data:
        return []
    
    questions = []
    selected_items = random.sample(data, min(len(data), num_questions * 2))
    
    for item in selected_items[:num_questions]:
        # Prendre la réponse et mélanger les mots
        answer = item["output"]
        words = answer.split()
        
        if len(words) < 3:
            continue
        
        # Mélanger les mots
        shuffled_words = words.copy()
        random.shuffle(shuffled_words)
        
        # Créer une question
        question = {
            "question": item["instruction"],
            "context": item["input"],
            "shuffled_words": shuffled_words,
            "correct_order": words
        }
        
        questions.append(question)
    
    return questions

# Afficher l'exercice en fonction du type sélectionné
if qa_data:
    st.subheader(f"{level} {topic} Exercises: {exercise_type}")
    
    if exercise_type == "Multiple Choice":
        questions = generate_multiple_choice(qa_data, num_questions, level)
        
        if questions:
            # Système de score
            if "mc_score" not in st.session_state:
                st.session_state.mc_score = 0
            if "mc_total" not in st.session_state:
                st.session_state.mc_total = 0
            
            for i, question in enumerate(questions):
                st.markdown(f"**Question {i+1}**: {question['question']}")
                
                if question["context"]:
                    st.markdown(f"*Context: {question['context']}*")
                
                # Créer un ID unique pour chaque question
                question_id = f"mc_{i}"
                
                # Utilisez st.radio pour les choix multiples
                answer = st.radio(
                    "Select your answer:",
                    question["options"],
                    key=question_id
                )
                
                # Vérifiez la réponse si le bouton est pressé
                if st.button("Check Answer", key=f"check_{i}"):
                    if answer == question["correct_answer"]:
                        st.success("Correct! 🎉")
                        st.session_state.mc_score += 1
                    else:
                        st.error(f"Incorrect. The correct answer is: {question['correct_answer']}")
                    
                    st.session_state.mc_total += 1
                
                st.divider()
            
            # Afficher le score total
            if st.session_state.mc_total > 0:
                st.metric("Your Score", f"{st.session_state.mc_score}/{st.session_state.mc_total}")
                
                # Réinitialiser le score
                if st.button("Reset Score"):
                    st.session_state.mc_score = 0
                    st.session_state.mc_total = 0
                    st.experimental_rerun()
        
        else:
            st.warning("Could not generate questions. Please try different settings.")
    
    elif exercise_type == "Fill in the Blanks":
        questions = generate_fill_in_blanks(qa_data, num_questions, level)
        
        if questions:
            # Système de score
            if "fb_score" not in st.session_state:
                st.session_state.fb_score = 0
            if "fb_total" not in st.session_state:
                st.session_state.fb_total = 0
            
            for i, question in enumerate(questions):
                st.markdown(f"**Question {i+1}**: {question['question']}")
                
                if question["context"]:
                    st.markdown(f"*Context: {question['context']}*")
                
                st.markdown(f"Complete the sentence: **{question['sentence']}**")
                
                # Créer des champs de texte pour les mots manquants
                user_answers = []
                for j, word in enumerate(question["missing_words"]):
                    user_answer = st.text_input(f"Blank {j+1}", key=f"blank_{i}_{j}")
                    user_answers.append(user_answer.strip())
                
                # Vérifiez la réponse si le bouton est pressé
                if st.button("Check Answer", key=f"check_fb_{i}"):
                    all_correct = True
                    for j, (user_ans, correct_word) in enumerate(zip(user_answers, question["missing_words"])):
                        if user_ans.lower() == correct_word.lower():
                            st.success(f"Blank {j+1}: Correct! ✓")
                        else:
                            st.error(f"Blank {j+1}: Incorrect. The correct answer is: {correct_word}")
                            all_correct = False
                    
                    if all_correct:
                        st.success("All correct! 🎉")
                        st.session_state.fb_score += 1
                    
                    st.session_state.fb_total += 1
                
                st.divider()
            
            # Afficher le score total
            if st.session_state.fb_total > 0:
                st.metric("Your Score", f"{st.session_state.fb_score}/{st.session_state.fb_total}")
                
                # Réinitialiser le score
                if st.button("Reset Score"):
                    st.session_state.fb_score = 0
                    st.session_state.fb_total = 0
                    st.experimental_rerun()
        
        else:
            st.warning("Could not generate fill-in-the-blank exercises. Please try different settings.")
    
    elif exercise_type == "Matching Pairs":
        pairs = generate_matching_pairs(qa_data, num_questions, level)
        
        if pairs and pairs["instructions"]:
            st.markdown("Match each question with its correct answer. Drag and drop or select from the dropdown.")
            
            # Système de score
            if "mp_score" not in st.session_state:
                st.session_state.mp_score = 0
            
            # Créer deux colonnes
            col1, col2 = st.columns(2)
            
            # Afficher les instructions
            with col1:
                st.subheader("Questions")
                for i, instruction in enumerate(pairs["instructions"]):
                    st.markdown(f"{i+1}. {instruction}")
            
            # Afficher les options de réponse avec des sélecteurs
            with col2:
                st.subheader("Answers")
                user_selections = []
                
                for i in range(len(pairs["instructions"])):
                    # Créer une liste de réponses mélangées
                    selection = st.selectbox(
                        f"Match for question {i+1}:",
                        pairs["shuffled_outputs"],
                        key=f"match_{i}"
                    )
                    user_selections.append(selection)
            
            # Vérifier les correspondances
            if st.button("Check Matching"):
                correct_count = 0
                for i, (user_selection, correct_output) in enumerate(zip(user_selections, pairs["outputs"])):
                    if user_selection == correct_output:
                        st.success(f"Question {i+1}: Correct! ✓")
                        correct_count += 1
                    else:
                        st.error(f"Question {i+1}: Incorrect. The correct answer is: {correct_output}")
                
                # Mettre à jour le score
                st.session_state.mp_score = correct_count
                
                # Afficher le score total
                st.metric("Your Score", f"{st.session_state.mp_score}/{len(pairs['instructions'])}")
            
            # Réinitialiser le score
            if st.button("Reset"):
                st.session_state.mp_score = 0
                st.experimental_rerun()
        
        else:
            st.warning("Could not generate matching pairs. Please try different settings.")
    
    elif exercise_type == "Flashcards":
        flashcards = generate_flashcards(qa_data, num_questions, level)
        
        if flashcards:
            st.markdown("Review these flashcards to test your knowledge. Click 'Flip' to see the answer.")
            
            # Tracker pour la carte actuelle
            if "current_card" not in st.session_state:
                st.session_state.current_card = 0
            
            # Tracker pour l'état de la carte (face ou dos)
            if "card_flipped" not in st.session_state:
                st.session_state.card_flipped = False
            
            # Navigation entre les cartes
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if st.button("Previous") and st.session_state.current_card > 0:
                    st.session_state.current_card -= 1
                    st.session_state.card_flipped = False
                    st.experimental_rerun()
            
            with col3:
                if st.button("Next") and st.session_state.current_card < len(flashcards) - 1:
                    st.session_state.current_card += 1
                    st.session_state.card_flipped = False
                    st.experimental_rerun()
            
            # Afficher la carte actuelle
            current_flashcard = flashcards[st.session_state.current_card]
            
            with st.container():
                st.markdown(f"### Card {st.session_state.current_card + 1}/{len(flashcards)}")
                
                # Créer un style de carte
                card_style = """
                <style>
                .flashcard {
                    padding: 2rem;
                    background-color: #f0f2f6;
                    border-radius: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    min-height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                    margin-bottom: 1rem;
                }
                </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)
                
                # Afficher le contenu de la carte
                if not st.session_state.card_flipped:
                    st.markdown(f'<div class="flashcard"><h2>{current_flashcard["front"]}</h2></div>', unsafe_allow_html=True)
                    if current_flashcard["context"]:
                        st.markdown(f"*Context: {current_flashcard['context']}*")
                else:
                    st.markdown(f'<div class="flashcard"><h2>{current_flashcard["back"]}</h2></div>', unsafe_allow_html=True)
                
                # Bouton pour retourner la carte
                if st.button("Flip Card"):
                    st.session_state.card_flipped = not st.session_state.card_flipped
                    st.experimental_rerun()
            
            # Indicateur de progression
            progress = (st.session_state.current_card + 1) / len(flashcards)
            st.progress(progress)
        
        else:
            st.warning("Could not generate flashcards. Please try different settings.")
    
    elif exercise_type == "Word Order":
        questions = generate_word_order(qa_data, num_questions, level)
        
        if questions:
            # Système de score
            if "wo_score" not in st.session_state:
                st.session_state.wo_score = 0
            if "wo_total" not in st.session_state:
                st.session_state.wo_total = 0
            
            st.markdown("Put the words in the correct order to form a sentence.")
            
            for i, question in enumerate(questions):
                st.markdown(f"**Question {i+1}**: {question['question']}")
                
                if question["context"]:
                    st.markdown(f"*Context: {question['context']}*")
                
                # Afficher les mots mélangés
                st.markdown("**Words to arrange:**")
                word_cols = st.columns(len(question["shuffled_words"]))
                
                for j, (col, word) in enumerate(zip(word_cols, question["shuffled_words"])):
                    with col:
                        st.markdown(f"**{word}**")
                
                # Champ pour la réponse de l'utilisateur
                user_answer = st.text_input(
                    "Type the correct sentence:",
                    key=f"wo_answer_{i}"
                )
                
                # Vérifiez la réponse si le bouton est pressé
                if st.button("Check Answer", key=f"check_wo_{i}"):
                    # Normaliser les réponses pour la comparaison (supprimer les espaces supplémentaires)
                    correct_answer = " ".join(question["correct_order"]).strip()
                    user_answer = user_answer.strip()
                    
                    if user_answer.lower() == correct_answer.lower():
                        st.success("Correct! 🎉")
                        st.session_state.wo_score += 1
                    else:
                        st.error(f"Incorrect. The correct answer is: {correct_answer}")
                    
                    st.session_state.wo_total += 1
                
                st.divider()
            
            # Afficher le score total
            if st.session_state.wo_total > 0:
                st.metric("Your Score", f"{st.session_state.wo_score}/{st.session_state.wo_total}")
                
                # Réinitialiser le score
                if st.button("Reset Score"):
                    st.session_state.wo_score = 0
                    st.session_state.wo_total = 0
                    st.experimental_rerun()
        
        else:
            st.warning("Could not generate word order exercises. Please try different settings.")

else:
    st.warning("No data available for creating exercises. Please check your data source.")

# Ajouter un élément gamifié
st.sidebar.markdown("---")
st.sidebar.subheader("Your Progress")

# Simuler des statistiques d'apprentissage
if "stats" not in st.session_state:
    st.session_state.stats = {
        "exercises_completed": 0,
        "correct_answers": 0,
        "streak_days": random.randint(1, 7),
        "exp_points": random.randint(100, 500)
    }

# Mettre à jour les statistiques
total_score = sum([
    getattr(st.session_state, "mc_score", 0),
    getattr(st.session_state, "fb_score", 0),
    getattr(st.session_state, "mp_score", 0),
    getattr(st.session_state, "wo_score", 0)
])
total_attempts = sum([
    getattr(st.session_state, "mc_total", 0),
    getattr(st.session_state, "fb_total", 0),
    getattr(st.session_state, "wo_total", 0)
])

if total_attempts > 0:
    st.session_state.stats["exercises_completed"] += total_attempts
    st.session_state.stats["correct_answers"] += total_score
    st.session_state.stats["exp_points"] += total_score * 10

# Afficher les statistiques
st.sidebar.metric("Exercises Completed", st.session_state.stats["exercises_completed"])
st.sidebar.metric("Correct Answers", st.session_state.stats["correct_answers"])
st.sidebar.metric("Day Streak", f"{st.session_state.stats['streak_days']} 🔥")
st.sidebar.metric("Experience Points", f"{st.session_state.stats['exp_points']} XP")

# Afficher le niveau
exp_points = st.session_state.stats["exp_points"]
if exp_points < 200:
    level_name = "Novice"
    progress = exp_points / 200
elif exp_points < 500:
    level_name = "Apprentice"
    progress = (exp_points - 200) / 300
elif exp_points < 1000:
    level_name = "Scholar"
    progress = (exp_points - 500) / 500
else:
    level_name = "Master"
    progress = 1.0

st.sidebar.markdown(f"**Level: {level_name}**")
st.sidebar.progress(progress)