#%%
# @title Setup HuggingFace Authentication
from huggingface_hub import login
HUGGINGFACE_TOKEN = "your_token_here"  # @param {type:"string"}
login(token=HUGGINGFACE_TOKEN)

#%%
HUGGINGFACE_TOKEN="your_token_here"

#%%
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from datasets import load_dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv

class LlamaFineTuner:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """Initialize Llama model for fine-tuning"""
        self.setup_auth()
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
    def load_model(self):
        """Load model and tokenizer"""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Model loaded on {self.device}")
        
    def prepare_dataset(self, dataset_name="your_dataset"):
        """Load and prepare dataset for fine-tuning"""
        dataset = load_dataset(dataset_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        self.tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
    def train(self, output_dir="./finetuned_model"):
        """Fine-tune the model"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=1000,
            save_total_limit=2,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        trainer.save_model()

def main():
    # Initialize fine-tuner
    tuner = LlamaFineTuner()
    
    # Prepare dataset and train
    tuner.prepare_dataset()
    tuner.train()
    
    # Test generation
    prompt = "Once upon a time"
    inputs = tuner.tokenizer(prompt, return_tensors="pt").to(tuner.device)
    outputs = tuner.model.generate(inputs.input_ids, max_length=100)
    print(tuner.tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()