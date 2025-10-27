"""
Page de cours et résumés pour l'application Streamlit.
"""

import streamlit as st
import pandas as pd
import os
import json
import sys

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

# Configuration de la page
st.set_page_config(
    page_title="Course Summaries",
    page_icon="📚",
    layout="wide",
)

# Appliquer le CSS personnalisé
st.markdown(utils.custom_css(), unsafe_allow_html=True)

# Titre de la page
st.title("📚 Course Summaries")
st.markdown("Review key concepts and summaries from English learning materials.")

# Fonction pour charger les données des cours depuis le CSV
@st.cache_data
def load_course_data():
    try:
        df = pd.read_csv("../data/data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading course data: {e}")
        return pd.DataFrame(columns=["pdf_name", "content", "clean_text"])

# Charger les données
df = load_course_data()

# Sidebar pour filtrer par niveau/manuel
with st.sidebar:
    st.subheader("Filter Content")
    
    # Extraire les noms uniques de PDF (correspondant aux niveaux/manuels)
    pdf_names = df["pdf_name"].unique() if not df.empty else []
    selected_pdf = st.selectbox("Select Textbook/Level", pdf_names)
    
    # Sélection du niveau de l'élève
    st.subheader("Student Level")
    student_level = st.select_slider(
        "Adapt content to your level:",
        options=["Beginner", "Elementary", "Intermediate", "Advanced", "Proficient"]
    )
    
    # Explication du niveau
    level_descriptions = {
        "Beginner": "Simplified summaries with basic vocabulary",
        "Elementary": "Clear explanations with common words",
        "Intermediate": "Standard summaries with regular vocabulary",
        "Advanced": "Detailed summaries with academic language",
        "Proficient": "Complete explanations with technical terminology"
    }
    st.info(level_descriptions[student_level])

# Fonction pour extraire les sections des contenus
def extract_sections(text):
    import re
    
    # Pattern pour détecter les titres (texte en majuscules suivi d'un saut de ligne)
    title_pattern = r'\b([A-Z][A-Z\s]+[A-Z])\b'
    
    # Trouver tous les titres potentiels
    titles = re.findall(title_pattern, text)
    
    # Filtrer les titres trop courts
    titles = [title for title in titles if len(title) > 5]
    
    # Retourner les titres uniques
    return list(set(titles))

# Fonction pour créer un résumé adapté au niveau
def create_adapted_summary(text, level):
    # Pour une application réelle, cette fonction pourrait utiliser le modèle pour 
    # générer des résumés adaptés au niveau de l'élève
    
    # Version simplifiée pour la démonstration
    summary = utils.summarize_course(text, level)
    
    # Ajuster la longueur en fonction du niveau
    max_chars = {
        "Beginner": 500,
        "Elementary": 800,
        "Intermediate": 1200,
        "Advanced": 2000,
        "Proficient": 3000
    }
    
    if len(summary) > max_chars[level]:
        return summary[:max_chars[level]] + "..."
    
    return summary

# Afficher le contenu filtré
if not df.empty and selected_pdf:
    # Filtrer les données par PDF sélectionné
    filtered_df = df[df["pdf_name"] == selected_pdf]
    
    if not filtered_df.empty:
        # Extraire le texte nettoyé du premier enregistrement correspondant
        content = filtered_df.iloc[0]["clean_text"] if "clean_text" in filtered_df.columns else filtered_df.iloc[0]["content"]
        
        # Extraire les sections
        sections = extract_sections(content)
        
        # Créer des onglets pour les différentes sections
        if sections:
            tabs = st.tabs(["Overview"] + sections[:5])  # Limiter à 5 sections pour éviter trop d'onglets
            
            # Onglet de résumé général
            with tabs[0]:
                st.subheader("Course Overview")
                adapted_summary = create_adapted_summary(content, student_level)
                st.markdown(adapted_summary)
                
                # Afficher des mots clés
                st.subheader("Key Vocabulary")
                col1, col2, col3 = st.columns(3)
                keywords = ["vocabulary", "grammar", "pronunciation", "reading", "writing", "listening", "speaking"]
                for i, keyword in enumerate(keywords):
                    with [col1, col2, col3][i % 3]:
                        st.markdown(f"- **{keyword.capitalize()}**")
            
            # Onglets pour chaque section
            for i, (tab, section) in enumerate(zip(tabs[1:], sections[:5])):
                with tab:
                    st.subheader(section)
                    
                    # Extraire le contenu de cette section
                    section_start = content.find(section)
                    next_section_start = float('inf')
                    for s in sections:
                        pos = content.find(s, section_start + len(section))
                        if pos > section_start and pos < next_section_start:
                            next_section_start = pos
                    
                    section_content = content[section_start:next_section_start] if next_section_start < float('inf') else content[section_start:]
                    
                    # Afficher le résumé adapté
                    adapted_section = create_adapted_summary(section_content, student_level)
                    st.markdown(adapted_section)
                    
                    # Ajouter une fonctionnalité interactive
                    with st.expander("Practice Exercises"):
                        exercise = utils.generate_exercise(section, student_level)
                        st.subheader(exercise["title"])
                        for i, question in enumerate(exercise["questions"]):
                            st.text(f"{i+1}. {question}")
                        
                        # Bouton pour afficher les réponses
                        if st.button("Show Answers", key=f"answers_{section}"):
                            for i, answer in enumerate(exercise["answers"]):
                                st.success(f"{i+1}. {answer}")
        else:
            # Si pas de sections détectées
            st.subheader("Course Content")
            adapted_summary = create_adapted_summary(content, student_level)
            st.markdown(adapted_summary)
    else:
        st.warning("No content found for the selected textbook.")
else:
    st.info("Please select a textbook from the sidebar to view its content.")