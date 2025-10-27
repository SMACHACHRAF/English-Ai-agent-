"""
Script d'entraînement du modèle LLM (Mistral) pour le chatbot d'apprentissage de l'anglais.
Ce script utilise PEFT (Parameter-Efficient Fine-Tuning) avec LoRA (Low-Rank Adaptation)
pour une adaptation efficace du modèle avec peu de ressources computationnelles.

PEFT permet de fine-tuner des LLMs de plusieurs milliards de paramètres en n'entraînant
qu'une petite fraction des paramètres (typiquement <1%), ce qui:
- Réduit drastiquement les besoins en mémoire GPU
- Accélère l'entraînement jusqu'à 10x
- Permet de stocker des adaptateurs légers (~quelques Mo) au lieu du modèle entier (plusieurs Go)
"""

import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model on QA dataset")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                      help="Base model to fine-tune (default: mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--output_dir", type=str, default="../models",
                      help="Directory to save the model (default: ../models)")
    parser.add_argument("--train_file", type=str, default="../data/train.json",
                      help="Path to the training data (default: ../data/train.json)")
    parser.add_argument("--val_file", type=str, default="../data/val.json",
                      help="Path to the validation data (default: ../data/val.json)")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-5,
                      help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length (default: 512)")
    parser.add_argument("--lora_r", type=int, default=8,
                      help="Rank of the LoRA adapters (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                      help="Alpha parameter for LoRA (default: 16)")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """
    Charge le modèle et le tokenizer avec les configurations appropriées pour le fine-tuning.
    """
    # Configuration pour la quantification 4 bits avec BitsAndBytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Charger le modèle quantifié
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def create_lora_config(args):
    """
    Crée la configuration LoRA pour le fine-tuning efficace.
    
    LoRA (Low-Rank Adaptation) est une méthode PEFT (Parameter-Efficient Fine-Tuning) qui:
    - Réduit le nombre de paramètres à entraîner (moins de mémoire requise)
    - Accélère l'entraînement et l'inférence
    - Permet de stocker plusieurs adaptations différentes pour un même modèle de base
    """
    return LoraConfig(
        r=args.lora_r,                # Rang de la décomposition (plus petit = moins de paramètres)
        lora_alpha=args.lora_alpha,   # Scaling factor pour le produit de matrices de rang faible
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Modules à adapter
        lora_dropout=0.05,            # Dropout dans les couches LoRA
        bias="none",                  # Pas de bias dans les adaptateurs
        task_type="CAUSAL_LM",        # Type de tâche (génération de texte)
        inference_mode=False,         # Mode entraînement
        fan_in_fan_out=False          # Paramètre spécifique pour certains types de couches
    )

def train(args):
    """
    Entraîne le modèle sur les données de questions-réponses en utilisant PEFT avec LoRA.
    PEFT permet d'entraîner efficacement de grands modèles de langage avec peu de ressources.
    """
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args)
    
    print("Loading dataset...")
    # Charger les données d'entraînement et de validation
    train_dataset = load_dataset("json", data_files=args.train_file)["train"]
    val_dataset = load_dataset("json", data_files=args.val_file)["train"]
    
    print("Creating LoRA configuration for PEFT...")
    # Créer la configuration LoRA pour le Parameter-Efficient Fine-Tuning
    peft_config = create_lora_config(args)
    
    print(f"Dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Configurer les arguments d'entraînement optimisés pour PEFT
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,             # Nombre d'époques d'entraînement
        per_device_train_batch_size=args.batch_size,  # Taille du batch par GPU
        per_device_eval_batch_size=args.batch_size,  # Taille du batch pour évaluation
        gradient_accumulation_steps=4,            # Permet de simuler de plus grands batch sizes
        optim="adamw_torch",                      # Optimiseur standard compatible
        save_strategy="steps",                    # Stratégie de sauvegarde
        save_steps=50,                            # Fréquence de sauvegarde du modèle
        logging_steps=10,                         # Fréquence des logs
        learning_rate=args.lr,                    # Taux d'apprentissage
        weight_decay=0.001,                       # Régularisation L2
        fp16=torch.cuda.is_available(),           # Utiliser la précision mixte si GPU disponible
        bf16=False,                               # Ne pas utiliser bfloat16
        max_grad_norm=0.3,                        # Clipping du gradient pour stabilité
        max_steps=-1,                             # Pas de limite d'étapes (utiliser les époques)
        warmup_ratio=0.03,                        # Période d'échauffement (3% de l'entraînement)
        group_by_length=True,                     # Regrouper les exemples de longueur similaire
        lr_scheduler_type="cosine",               # Décroissance du LR en cosinus
        eval_strategy="steps",                    # Stratégie d'évaluation (au lieu de evaluation_strategy)
        eval_steps=50,                            # Fréquence d'évaluation
        load_best_model_at_end=True,              # Charger le meilleur modèle à la fin
        report_to="tensorboard",                  # Logger pour visualisation
        push_to_hub=False,                        # Ne pas uploader automatiquement
    )
    
    # Formatter les prompts pour le modèle - format d'instruction spécifique à Mistral
    def formatting_prompts_func(examples):
        """
        Convertit les exemples d'entraînement au format d'instruction pour Mistral.
        Format: [INST] question + contexte [/INST] réponse
        """
        instructions = examples["instruction"]  # Questions
        inputs = examples["input"]              # Contextes
        outputs = examples["output"]            # Réponses
        
        prompts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text.strip():
                # Format pour Mistral avec contexte
                prompt = f"<s>[INST] {instruction}\n\nContext: {input_text} [/INST] {output}</s>"
            else:
                # Format pour Mistral sans contexte
                prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
            prompts.append(prompt)
            
        return {"text": prompts}
    
    # Configurer le Supervised Fine-Tuning Trainer avec PEFT
    print("Setting up SFT Trainer with PEFT configuration...")
    
    # Afficher les paramètres pour débogage
    print(f"Model type: {type(model).__name__}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Max length: {args.max_length}")
    print(f"Dataset format - Column names: {train_dataset.column_names}")
    
    # Prétraiter les datasets pour ajouter une colonne "text"
    def apply_formatting_to_dataset(dataset):
        formatted = formatting_prompts_func(dataset)
        return dataset.add_column("text", formatted["text"])
    
    print("Preprocessing datasets...")
    train_dataset_processed = apply_formatting_to_dataset(train_dataset)
    val_dataset_processed = apply_formatting_to_dataset(val_dataset)
    
    print(f"Processed dataset columns: {train_dataset_processed.column_names}")
    
    # Initialiser le trainer avec les paramètres simplifiés
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=val_dataset_processed,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    
    print("Starting PEFT training with LoRA adapters...")
    # Entraîner le modèle avec les adaptateurs LoRA
    trainer.train()
    
    print(f"Training completed. Saving PEFT model to {args.output_dir}")
    # Sauvegarder le modèle (seuls les adaptateurs sont sauvegardés)
    trainer.save_model()
    
    # Message explicatif sur ce qui a été sauvegardé
    print("Note: Only the LoRA adapters have been saved, not the full model.")
    print("The adapters can be merged with the base model during inference for best performance.")

def test_model(args):
    """
    Teste le modèle LLM entraîné sur quelques exemples.
    """
    # Charger le modèle entraîné
    print("Loading model from:", args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Activer le mode évaluation
    model.eval()
    
    # Si le modèle est un PeftModel, fusionner les adaptateurs LoRA avec le modèle de base
    # Cela améliore les performances d'inférence en évitant le calcul séparé des adaptateurs
    if hasattr(model, "merge_and_unload"):
        print("Merging LoRA adapters with base model for faster inference...")
        model = model.merge_and_unload()
    
    # Exemples de test avec contextes
    test_examples = [
        {
            "question": "How do I introduce myself in English?",
            "context": "When meeting new people, it's important to know how to introduce yourself properly. Common phrases include 'Hello, my name is...' or 'Hi, I'm...'. You can also add information about where you're from or what you do."
        },
        {
            "question": "What's the difference between 'been' and 'gone'?",
            "context": "In English grammar, 'been' and 'gone' are both past participles but are used differently. 'Been' is used when someone went somewhere and returned: 'I have been to Paris three times.' 'Gone' is used when someone went somewhere and hasn't returned yet: 'She has gone to the store (and is still there).'"
        },
        {
            "question": "Can you explain the present perfect tense?",
            "context": "The present perfect tense in English is formed with 'have/has' + past participle. It's used to talk about actions that started in the past and continue to the present, completed actions with a result in the present, and experiences in an unspecified time in the past. Examples include 'I have lived here for five years' or 'She has visited Paris twice.'"
        }
    ]
    
    print("\n===== TESTING MODEL =====\n")
    
    for example in test_examples:
        question = example["question"]
        context = example["context"]
        
        # Formater le prompt pour la génération
        prompt = f"[INST] {question}\n\nContext: {context} [/INST]"
            
        # Tokeniser l'entrée
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Générer la réponse
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1
            )
        
        # Décoder la réponse générée
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraire uniquement la réponse générée (après le prompt)
        predicted_answer = generated_text[len(prompt):].strip()
        
        # Afficher les résultats
        print(f"\nQuestion: {question}")
        print(f"Context: {context}")
        print(f"Generated answer: {predicted_answer}")
        print("-" * 50)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set default values for missing arguments
    if not hasattr(args, 'max_length'):
        args.max_length = 512
    if not hasattr(args, 'batch_size'):
        args.batch_size = 4
    if not hasattr(args, 'lr'):
        args.lr = 2e-5
    
    # Set logging verbosity
    logging.set_verbosity_info()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train(args)
    
    # Test the model
    test_model(args)

if __name__ == "__main__":
    main()