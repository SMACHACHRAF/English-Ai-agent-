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
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import transformers
from transformers import Trainer, default_data_collator

def parse_args():
    # Obtenir le répertoire racine du projet (dossier DEEP)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    parser = argparse.ArgumentParser(description="Fine-tune a language model on QA dataset")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                      help="Base model to fine-tune (default: mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(root_dir, "models"),
                      help="Directory to save the model (default: DEEP/models)")
    parser.add_argument("--train_file", type=str, default=os.path.join(root_dir, "data", "train.json"),
                      help="Path to the training data (default: DEEP/data/train.json)")
    parser.add_argument("--val_file", type=str, default=os.path.join(root_dir, "data", "val.json"),
                      help="Path to the validation data (default: DEEP/data/val.json)")
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
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected, using CPU only. Training will be very slow.")
    
    # Configuration adaptée à l'environnement
    if torch.cuda.is_available():
        # Configuration pour la quantification 4 bits avec BitsAndBytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        device_map = "auto"
    else:
        # Configuration CPU seulement - modèle plus petit pour la compatibilité
        print("WARNING: Using CPU only. Switching to a smaller model: TinyLlama-1.1B")
        args.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        bnb_config = None
        device_map = None
    """# tinyllama_config.py
class TinyLlamaConfig:
    """
    """Configuration pour TinyLlama-1.1B"""
    """
    def __init__(self):
        # Vocabulaire et embeddings
        self.vocab_size = 32000
        self.hidden_size = 2048
        self.max_seq_len = 2048
        
        # Attention
        self.num_heads = 32
        self.head_dim = self.hidden_size // self.num_heads  # 64
        self.causal = True  # Masked self-attention
        
        # Feed Forward (SwiGLU)
        self.intermediate_size = 8192
        self.activation = "silu"
        
        # Normalisation
        self.rms_norm_eps = 1e-6
        
        # Dropout
        self.attention_dropout = 0.0
        self.ffn_dropout = 0.0
        
        # Décoder
        self.num_layers = 22
        
        # LM Head
        self.tie_word_embeddings = True  # LM Head shared with token embeddings
        
        # Positional Encoding
        self.use_rope = True  # Rotary Positional Embeddings
        
    def __repr__(self):
        return f"TinyLlamaConfig(hidden_size={self.hidden_size}, num_layers={self.num_layers}, num_heads={self.num_heads}, vocab_size={self.vocab_size})"

# Exemple d'utilisation
if _name_ == "_main_":
    config = TinyLlamaConfig()
    print(config)
tinyllama_config.py
Écrire à Projet IA
"""
    # Charger le modèle
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
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
    
    # Appliquer LoRA au modèle
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print(f"Dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Configurer les arguments d'entraînement optimisés pour l'environnement
    # Adapter les paramètres en fonction de la disponibilité du GPU
    if torch.cuda.is_available():
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=50,
            logging_steps=10,
            learning_rate=args.lr,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=50,
            load_best_model_at_end=True,
            report_to="none",
            push_to_hub=False,
        )
    else:
        # Configuration allégée pour CPU uniquement
        print("Setting up lightweight training configuration for CPU")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=1,                   # Réduire le nombre d'époques pour CPU
            per_device_train_batch_size=1,        # Réduire la taille du batch
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,        # Augmenter l'accumulation de gradient
            optim="adamw_torch",
            save_strategy="epoch",                # Sauvegarder seulement à la fin des époques
            logging_steps=5,
            learning_rate=5e-5,
            weight_decay=0.001,
            fp16=False,                           # Pas de précision mixte sur CPU
            bf16=False,
            max_grad_norm=1.0,
            max_steps=-1,
            warmup_ratio=0.0,                     # Pas de warmup
            group_by_length=False,                # Pas de regroupement
            lr_scheduler_type="linear",           # Scheduler plus simple
            eval_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
            push_to_hub=False,
        )
    
    # Formatter les prompts pour le modèle - format d'instruction spécifique à Mistral
    def formatting_func(example):
        """
        Convertit un exemple au format d'instruction pour Mistral.
        Format: [INST] question + contexte [/INST] réponse
        """
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        if input_text.strip():
            # Format pour Mistral avec contexte
            prompt = f"<s>[INST] {instruction}\n\nContext: {input_text} [/INST] {output}</s>"
        else:
            # Format pour Mistral sans contexte
            prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
            
        return {"text": prompt}
    
    # Prétraiter et tokeniser les datasets en une seule étape
    print("Preprocessing and tokenizing datasets...")
    
    def process_and_tokenize(examples):
        texts = []
        
        # Adapter le format du prompt au modèle
        if "TinyLlama" in args.model_name:
            # Format pour TinyLlama
            for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
                if input_text.strip():
                    prompt = f"<|system|>\nYou are a helpful English learning assistant.\n<|user|>\n{instruction}\nContext: {input_text}\n<|assistant|>\n{output}"
                else:
                    prompt = f"<|system|>\nYou are a helpful English learning assistant.\n<|user|>\n{instruction}\n<|assistant|>\n{output}"
                texts.append(prompt)
        else:
            # Format pour Mistral
            for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
                if input_text.strip():
                    prompt = f"<s>[INST] {instruction}\n\nContext: {input_text} [/INST] {output}</s>"
                else:
                    prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
                texts.append(prompt)
        
        # Tokeniser directement - réduire la longueur maximale pour économiser la mémoire
        max_length = min(args.max_length, 384)  # Limiter à 384 tokens pour réduire l'utilisation de mémoire
        tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        
        # Préparer les labels pour le causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Appliquer la transformation
    train_dataset = train_dataset.map(process_and_tokenize, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(process_and_tokenize, batched=True, remove_columns=val_dataset.column_names)
    
    print(f"Dataset columns after processing: {train_dataset.column_names}")
    
    # Configurer le trainer standard pour un maximum de compatibilité
    print("Setting up Trainer with PEFT...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    print("Starting PEFT training with LoRA adapters...")
    # Entraîner le modèle avec les adaptateurs LoRA
    trainer.train()
    
    print(f"Training completed. Saving PEFT model to {args.output_dir}")
    # Sauvegarder le modèle (seuls les adaptateurs sont sauvegardés)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Train model
    train(args)
    
    # Test model
    test_model(args)

if __name__ == "__main__":
    main()