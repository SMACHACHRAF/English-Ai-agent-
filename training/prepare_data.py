"""
Script de préparation des données pour le fine-tuning du modèle LLaMA.
Convertit les données au format approprié et les divise en ensembles d'entraînement et de validation.
"""

import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_qa_data(file_path):
    """
    Charge les données questions-réponses depuis un fichier JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} QA pairs from {file_path}")
    return data

def format_data_for_training(data):
    """
    Formate les données au format attendu par le modèle.
    """
    # Les données sont déjà au format {"instruction": "...", "input": "...", "output": "..."}
    return data

def split_data(data, test_size=0.1):
    """
    Divise les données en ensembles d'entraînement et de validation.
    """
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    print(f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")
    return train_data, val_data

def save_data(train_data, val_data, output_dir="../data"):
    """
    Sauvegarde les données d'entraînement et de validation au format JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved validation data to {val_path}")

def analyze_data(data):
    """
    Analyse les données pour obtenir des statistiques.
    """
    instruction_lengths = [len(item["instruction"].split()) for item in data]
    input_lengths = [len(item["input"].split()) for item in data]
    output_lengths = [len(item["output"].split()) for item in data]
    
    print("\nData Statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Average instruction length: {sum(instruction_lengths) / len(instruction_lengths):.1f} words")
    print(f"Average input length: {sum(input_lengths) / len(input_lengths):.1f} words")
    print(f"Average output length: {sum(output_lengths) / len(output_lengths):.1f} words")
    print(f"Min/Max instruction length: {min(instruction_lengths)}/{max(instruction_lengths)} words")
    print(f"Min/Max input length: {min(input_lengths)}/{max(input_lengths)} words")
    print(f"Min/Max output length: {min(output_lengths)}/{max(output_lengths)} words")

def main():
    # Chemins des fichiers
    input_file = "../data/Question_Reponse_DATA_converted.json"
    
    # Charger les données
    data = load_qa_data(input_file)
    
    # Analyser les données
    analyze_data(data)
    
    # Formater les données
    formatted_data = format_data_for_training(data)
    
    # Diviser les données
    train_data, val_data = split_data(formatted_data)
    
    # Sauvegarder les données
    save_data(train_data, val_data)

if __name__ == "__main__":
    main()