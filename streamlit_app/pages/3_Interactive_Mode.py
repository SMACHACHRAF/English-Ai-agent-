"""
Page pour le mode interactif par niveau pour l'application Streamlit.
"""

import streamlit as st
import pandas as pd
import os
import json
import random
import sys

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# Configuration de la page
st.set_page_config(
    page_title="Interactive Mode",
    page_icon="🎯",
    layout="wide",
)

# Appliquer le CSS personnalisé
st.markdown(utils.custom_css(), unsafe_allow_html=True)

# Titre de la page
st.title("🎯 Interactive Learning by Level")
st.markdown("A personalized learning experience based on your English proficiency level.")

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

# Sidebar pour la sélection du niveau
with st.sidebar:
    st.subheader("Your Profile")
    
    # Sélection du niveau
    level = st.select_slider(
        "Select your proficiency level:",
        options=["Beginner", "Elementary", "Intermediate", "Advanced", "Proficient"]
    )
    
    # Information sur le niveau sélectionné
    level_info = {
        "Beginner": "You're just starting with English basics. Focus on simple vocabulary and phrases.",
        "Elementary": "You know some English and can have basic conversations. Time to expand your vocabulary and grammar.",
        "Intermediate": "You can communicate in most situations. Let's work on fluency and more complex structures.",
        "Advanced": "You're comfortable with English. Let's refine your skills with nuanced language and complex topics.",
        "Proficient": "You have mastery of English. Let's perfect your skills with academic and professional language."
    }
    st.info(level_info[level])
    
    # Choisir un mode
    st.subheader("Learning Mode")
    mode = st.radio("Choose your learning mode:", ["Conversation", "Guided Learning", "Challenge"])

# Fonction pour obtenir des questions filtrées par niveau
def get_level_questions(data, level):
    # Cette fonction simule un filtrage par niveau
    # Dans une application réelle, les données seraient étiquetées par niveau
    
    # Simulation: assigner des niveaux aux questions en fonction de la complexité
    level_mapping = {
        "Beginner": lambda q: len(q["instruction"].split()) < 8 and "?" in q["instruction"],
        "Elementary": lambda q: 6 <= len(q["instruction"].split()) < 10,
        "Intermediate": lambda q: 8 <= len(q["instruction"].split()) < 15,
        "Advanced": lambda q: len(q["instruction"].split()) >= 12 or any(word in q["instruction"].lower() for word in ["explain", "difference", "compare", "analyze"]),
        "Proficient": lambda q: len(q["instruction"].split()) >= 15 or any(word in q["instruction"].lower() for word in ["evaluate", "critique", "synthesize", "assess"])
    }
    
    # Filtrer les questions pour le niveau sélectionné
    filtered = [q for q in data if level_mapping[level](q)]
    
    # Si pas assez de questions, prendre des questions de niveaux adjacents
    if len(filtered) < 5:
        if level == "Beginner":
            filtered.extend([q for q in data if level_mapping["Elementary"](q)][:10])
        elif level == "Proficient":
            filtered.extend([q for q in data if level_mapping["Advanced"](q)][:10])
        else:
            levels = ["Beginner", "Elementary", "Intermediate", "Advanced", "Proficient"]
            current_idx = levels.index(level)
            if current_idx > 0:  # Ajouter des questions du niveau inférieur
                filtered.extend([q for q in data if level_mapping[levels[current_idx-1]](q)][:5])
            if current_idx < len(levels) - 1:  # Ajouter des questions du niveau supérieur
                filtered.extend([q for q in data if level_mapping[levels[current_idx+1]](q)][:5])
    
    return filtered[:30]  # Limiter à 30 questions

# Fonction pour générer un parcours d'apprentissage guidé
def generate_learning_path(level):
    learning_paths = {
        "Beginner": [
            {"title": "Basic Greetings", "topics": ["Hello and Goodbye", "Introducing Yourself", "Basic Questions"]},
            {"title": "Numbers and Colors", "topics": ["Counting to 20", "Basic Colors", "Simple Descriptions"]},
            {"title": "Daily Routines", "topics": ["Telling Time", "Days of the Week", "Simple Present Tense"]}
        ],
        "Elementary": [
            {"title": "Describing People", "topics": ["Physical Appearance", "Personality Traits", "Clothing"]},
            {"title": "Around Town", "topics": ["Directions", "Public Places", "Transportation"]},
            {"title": "Food and Dining", "topics": ["Ordering Food", "Cooking Terms", "Food Preferences"]}
        ],
        "Intermediate": [
            {"title": "Travel and Tourism", "topics": ["Planning a Trip", "Cultural Differences", "Travel Problems"]},
            {"title": "Work and Career", "topics": ["Job Interviews", "Office Vocabulary", "Business Communication"]},
            {"title": "Entertainment", "topics": ["Movies and TV", "Music", "Events and Activities"]}
        ],
        "Advanced": [
            {"title": "Current Events", "topics": ["News Vocabulary", "Expressing Opinions", "Debates"]},
            {"title": "Literature and Arts", "topics": ["Book Reviews", "Art Appreciation", "Critical Analysis"]},
            {"title": "Science and Technology", "topics": ["Tech Terminology", "Scientific Discoveries", "Future Trends"]}
        ],
        "Proficient": [
            {"title": "Academic English", "topics": ["Research Papers", "Presentations", "Academic Discourse"]},
            {"title": "Professional Development", "topics": ["Negotiation", "Leadership", "Specialized Fields"]},
            {"title": "Cultural Nuances", "topics": ["Idioms and Slang", "Cultural References", "Humor and Irony"]}
        ]
    }
    
    return learning_paths[level]

# Mode de conversation
if mode == "Conversation":
    st.header("Conversation Mode")
    st.markdown(f"Practice English conversation at {level} level with our AI tutor.")
    
    # Initialiser l'état de conversation si ce n'est pas déjà fait
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    # Afficher les messages précédents
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie pour l'utilisateur
    if prompt := st.chat_input("Type your message here..."):
        # Ajouter le message utilisateur à l'historique
        st.session_state.conversation.append({"role": "user", "content": prompt})
        
        # Afficher le message utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer une réponse (simulée)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Connecter au modèle d'IA pour obtenir une réponse réelle
                try:
                    # Tenter d'utiliser notre modèle entraîné
                    response = utils.generate_ai_response(prompt, level)
                except Exception as e:
                    st.warning(f"Could not connect to AI model: {e}", icon="⚠️")
                    
                    # Fallback aux réponses prédéfinies si le modèle ne fonctionne pas
                    if "hello" in prompt.lower() or "hi" in prompt.lower():
                        response = f"Hello! 👋 Welcome to the {level} level English practice. How can I help you today?"
                    elif "how are you" in prompt.lower():
                        response = "I'm doing well, thank you for asking! How about you? How's your day going?"
                    elif "help" in prompt.lower() or "learn" in prompt.lower():
                        response = f"I'd be happy to help you learn English at the {level} level! Would you like to practice vocabulary, grammar, or have a conversation about a specific topic?"
                    elif "grammar" in prompt.lower():
                        if level == "Beginner":
                            response = "Let's start with basic grammar! The simple present tense is used to talk about habits, facts, and routines. For example: 'I eat breakfast every day.' Would you like to practice making some sentences?"
                        elif level in ["Elementary", "Intermediate"]:
                            response = "Let's work on the present perfect tense! It's used for actions that started in the past and continue to the present. For example: 'I have lived here for three years.' Can you create a sentence using this tense?"
                        else:
                            response = "At your advanced level, let's discuss conditional sentences. The third conditional is used for impossible situations in the past: 'If I had studied harder, I would have passed the exam.' Can you create your own example?"
                    elif "vocabulary" in prompt.lower():
                        if level == "Beginner":
                            response = "Let's learn some basic vocabulary! Here are some words about food: bread, water, apple, banana, chicken. Can you use these words in sentences?"
                        elif level in ["Elementary", "Intermediate"]:
                            response = "Let's expand your vocabulary! Here are some words to describe people: ambitious, considerate, reliable, enthusiastic, creative. Which of these words might describe you?"
                        else:
                            response = "For your advanced vocabulary practice, let's explore synonyms for 'important': crucial, essential, significant, paramount, imperative. Can you use each in a sentence with slightly different meanings?"
                    elif "what's the difference between" in prompt.lower() or "difference between" in prompt.lower():
                        if "been" in prompt.lower() and "gone" in prompt.lower():
                            response = "Great question! 'Been' and 'gone' are both past participles of 'go', but they're used differently:\n\n'Been' is used when someone went somewhere and has returned: 'I have been to Paris three times.'\n\n'Gone' is used when someone went somewhere and hasn't returned yet: 'She has gone to the store (and is still there).'"
                        else:
                            response = f"That's a good question about language differences! Could you provide more context or examples of what you're trying to understand? This would help me give you a more precise explanation at your {level} level."
                    else:
                        response = f"That's interesting! As your {level} level English tutor, I'd suggest we explore this topic further. Would you like to learn more vocabulary related to this, practice specific grammar points, or just continue our conversation?"
                
                st.markdown(response)
        
        # Ajouter la réponse à l'historique
        st.session_state.conversation.append({"role": "assistant", "content": response})

# Mode d'apprentissage guidé
elif mode == "Guided Learning":
    st.header("Guided Learning Path")
    st.markdown(f"Follow a structured learning program tailored to your {level} level.")
    
    # Générer un parcours d'apprentissage
    learning_path = generate_learning_path(level)
    
    # Afficher le parcours d'apprentissage
    for i, module in enumerate(learning_path):
        with st.expander(f"Module {i+1}: {module['title']}", expanded=(i==0)):
            st.markdown(f"**Learning Objectives:**")
            for topic in module["topics"]:
                st.markdown(f"- {topic}")
            
            st.markdown("---")
            
            # Simuler un contenu de leçon
            st.markdown("### Lesson Content")
            
            # Contenu simulé basé sur le niveau et le module
            if level == "Beginner" and "Basic Greetings" in module["title"]:
                st.markdown("""
                **Basic English Greetings**
                
                1. **Hello / Hi** - Use these to greet someone.
                   - *Example:* "Hello! How are you?"
                
                2. **Good morning / afternoon / evening** - Time-specific greetings.
                   - *Example:* "Good morning, John!"
                
                3. **My name is...** - Introducing yourself.
                   - *Example:* "Hello, my name is Sarah."
                
                4. **Nice to meet you** - Use when meeting someone for the first time.
                   - *Example:* "Nice to meet you, David!"
                """)
            elif level == "Intermediate" and "Travel and Tourism" in module["title"]:
                st.markdown("""
                **Planning a Trip in English**
                
                1. **Booking Accommodation**
                   - *Vocabulary:* reservation, availability, check-in, check-out
                   - *Example dialogue:* "I'd like to book a double room for three nights starting from May 15th."
                
                2. **Transportation Options**
                   - *Vocabulary:* round-trip, one-way, fare, schedule, delay
                   - *Phrase:* "Could you tell me how frequently the trains to London run?"
                
                3. **Asking for Recommendations**
                   - *Structure:* "Can you recommend any [restaurants/museums/parks] near [location]?"
                   - *Example:* "Are there any must-see attractions in this area?"
                """)
            else:
                st.markdown(f"Content for {module['title']} at {level} level will be available here.")
            
            # Ajouter un exercice interactif
            st.markdown("### Practice Exercise")
            
            # Exercice basé sur le niveau
            if "Beginner" in level:
                question = "Complete the dialogue: A: Hello, _____ is your name? B: My name is John."
                answer = "what"
                user_answer = st.text_input("Your answer:", key=f"ex_{i}")
                
                if st.button("Check", key=f"check_{i}"):
                    if user_answer.lower() == answer:
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Not quite. The correct answer is: '{answer}'")
            elif "Elementary" in level:
                question = "Put the words in the correct order: yesterday / to the museum / went / I"
                answer = "I went to the museum yesterday"
                user_answer = st.text_input("Your answer:", key=f"ex_{i}")
                
                if st.button("Check", key=f"check_{i}"):
                    if user_answer.lower() == answer.lower():
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Not quite. The correct answer is: '{answer}'")
            else:
                options = ["to discuss", "discussing", "discuss", "for discussing"]
                question = "Choose the correct form: I'm looking forward _______ the project with you."
                answer = "to discussing"
                user_answer = st.radio("Select your answer:", options, key=f"ex_{i}")
                
                if st.button("Check", key=f"check_{i}"):
                    if user_answer == answer:
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Not quite. The correct answer is: '{answer}'")
            
            st.markdown("---")
            st.markdown("**Ready for more? Click the button below to take a quiz on this module.**")
            
            if st.button("Take Module Quiz", key=f"quiz_{i}"):
                st.success("Quiz feature will be available in the full version!")

# Mode challenge
elif mode == "Challenge":
    st.header("Challenge Mode")
    st.markdown(f"Test your {level} level English skills with these challenges.")
    
    # Obtenez des questions filtrées par niveau
    level_questions = get_level_questions(qa_data, level) if qa_data else []
    
    if level_questions:
        # Sélectionner 3 questions aléatoires pour le défi
        challenge_questions = random.sample(level_questions, min(3, len(level_questions)))
        
        st.subheader("Today's Challenge")
        st.markdown(f"Complete these {level} level questions to earn points!")
        
        # Initialiser le score de défi si ce n'est pas déjà fait
        if "challenge_score" not in st.session_state:
            st.session_state.challenge_score = 0
        if "challenge_completed" not in st.session_state:
            st.session_state.challenge_completed = False
        
        # Afficher un compteur de temps
        if "challenge_time" not in st.session_state:
            st.session_state.challenge_time = 120  # 2 minutes
        
        if not st.session_state.challenge_completed:
            # Afficher le temps restant
            st.markdown(f"**Time Remaining:** {st.session_state.challenge_time} seconds")
            st.progress(st.session_state.challenge_time / 120)
            
            # Afficher les questions
            for i, question in enumerate(challenge_questions):
                st.markdown(f"**Question {i+1}:** {question['instruction']}")
                
                if question["input"]:
                    st.markdown(f"*Context: {question['input']}*")
                
                # Champ de réponse
                user_answer = st.text_area(f"Your answer for question {i+1}:", key=f"challenge_{i}")
                
                # Comparer la réponse (de manière très basique)
                if len(user_answer) > 0:
                    similarity = len(set(user_answer.lower().split()) & set(question["output"].lower().split())) / max(len(set(user_answer.lower().split())), len(set(question["output"].lower().split())))
                    if similarity > 0.6:  # Seuil de similarité simple
                        st.session_state.challenge_score += 1
                
                st.divider()
            
            # Bouton pour soumettre le défi
            if st.button("Submit Challenge"):
                st.session_state.challenge_completed = True
                st.experimental_rerun()
        else:
            # Afficher les résultats
            st.subheader("Challenge Results")
            st.metric("Your Score", f"{st.session_state.challenge_score} / {len(challenge_questions)}")
            
            # Afficher le feedback
            if st.session_state.challenge_score == len(challenge_questions):
                st.success("Perfect score! 🏆 Your English skills at this level are excellent!")
            elif st.session_state.challenge_score >= len(challenge_questions) * 0.7:
                st.success("Great job! 🌟 You have a good command of English at this level.")
            elif st.session_state.challenge_score >= len(challenge_questions) * 0.4:
                st.warning("Good effort! 👍 Keep practicing to improve your skills at this level.")
            else:
                st.error("You might want to review the basics of this level. Don't worry, practice makes perfect! 💪")
            
            # Afficher les réponses correctes
            st.subheader("Correct Answers")
            for i, question in enumerate(challenge_questions):
                with st.expander(f"Question {i+1}: {question['instruction']}"):
                    st.markdown(f"**Correct Answer:** {question['output']}")
            
            # Bouton pour réinitialiser le défi
            if st.button("Try Another Challenge"):
                st.session_state.challenge_score = 0
                st.session_state.challenge_completed = False
                st.session_state.challenge_time = 120
                st.experimental_rerun()
    else:
        st.warning("No challenge questions available for your level. Please try a different level or check the data source.")

# Afficher les statistiques d'apprentissage dans la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Learning Statistics")

# Initialiser les statistiques si elles n'existent pas déjà
if "learning_stats" not in st.session_state:
    st.session_state.learning_stats = {
        "time_spent": 0,
        "challenges_completed": 0,
        "correct_answers": 0,
        "vocab_learned": 0
    }

# Mettre à jour les statistiques en fonction des activités
if "challenge_completed" in st.session_state and st.session_state.challenge_completed:
    st.session_state.learning_stats["challenges_completed"] += 1
    st.session_state.learning_stats["correct_answers"] += st.session_state.challenge_score
    st.session_state.learning_stats["vocab_learned"] += 5  # Estimer 5 nouveaux mots par défi

# Afficher les statistiques
st.sidebar.metric("Time Spent Learning", f"{st.session_state.learning_stats['time_spent']} mins")
st.sidebar.metric("Challenges Completed", st.session_state.learning_stats["challenges_completed"])
st.sidebar.metric("Correct Answers", st.session_state.learning_stats["correct_answers"])
st.sidebar.metric("Vocabulary Learned", st.session_state.learning_stats["vocab_learned"])

# Visualisation des progrès
if st.sidebar.checkbox("Show Progress Chart"):
    # Dans une application réelle, ces données seraient stockées et récupérées
    # Ici, nous utilisons des données fictives pour la démonstration
    user_history = [
        {"date": "2025-10-15", "level": "Beginner"},
        {"date": "2025-10-17", "level": "Beginner"},
        {"date": "2025-10-19", "level": "Elementary"},
        {"date": "2025-10-21", "level": level}
    ]
    
    progress_chart = utils.create_level_progress_chart(user_history)
    if progress_chart:
        st.sidebar.pyplot(progress_chart)