from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import re
import transformers
from transformers import Trainer
import peft
import torch
import bitsandbytes as bnb
from datasets import Dataset

# Load environment variables
load_dotenv()

class HarryPotterExample(BaseModel):
    """Schema for Harry Potter training examples"""
    context: str = Field(description="The context or scene from Harry Potter")
    question: str = Field(description="A question about the context")
    answer: str = Field(description="The answer to the question")
    book: str = Field(description="The Harry Potter book this example is from")

class HarryPotterFineTuner:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        """Initialize the fine-tuning process for Harry Potter."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        )
        
        # Prepare model for training
        self.model = peft.prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = peft.LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM
        )
        
        self.model = peft.get_peft_model(self.model, lora_config)
        self.training_data = []

    def clean_text(self, text: str) -> str:
        """Clean the text by removing special characters and extra whitespace."""
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = ' '.join(text.split())
        return text

    def create_qa_pair(self, text: str, book_name: str) -> Dict:
        """Generate a single QA pair from text."""
        try:
            # Generate question
            prompt = f"Create a question about this Harry Potter text:\n{text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            question = self.tokenizer.decode(
                self.model.generate(**inputs, max_length=100, temperature=0.7)[0],
                skip_special_tokens=True
            )
            
            # Generate answer
            answer_prompt = f"Answer this question about Harry Potter:\nQuestion: {question}\nContext: {text}"
            inputs = self.tokenizer(answer_prompt, return_tensors="pt").to(self.device)
            answer = self.tokenizer.decode(
                self.model.generate(**inputs, max_length=200, temperature=0.7)[0],
                skip_special_tokens=True
            )
            
            return {
                "context": text,
                "question": question,
                "answer": answer,
                "book": book_name
            }
        except Exception as e:
            print(f"Error creating QA pair: {e}")
            return None

    def load_books(self, directory_path: str) -> None:
        """Load and process Harry Potter books."""
        try:
            for doc in DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader).load():
                book_name = os.path.basename(doc.metadata["source"])
                print(f"\nProcessing: {book_name}")
                
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                ).split_documents([doc])
                
                for chunk in tqdm(chunks, desc=f"Processing {book_name}"):
                    if qa_pair := self.create_qa_pair(chunk.page_content, book_name):
                        self.training_data.append(qa_pair)

            print(f"\nLoaded {len(self.training_data)} training examples")
        except Exception as e:
            print(f"Error loading books: {e}")
            raise

    def prepare_dataset(self) -> Dataset:
        """Prepare training dataset."""
        formatted_data = []
        for example in self.training_data:
            text = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answer']}"
            tokenized = self.tokenizer(text, truncation=True, max_length=512, padding="max_length")
            formatted_data.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            })
        return Dataset.from_list(formatted_data)

    def fine_tune(self, output_dir: str = "harry_potter_fine_tuned") -> None:
        """Fine-tune the model."""
        try:
            trainer = Trainer(
                model=self.model,
                args=transformers.TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=3,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    learning_rate=2e-4,
                    weight_decay=0.01,
                    warmup_steps=100,
                    logging_steps=10,
                    save_strategy="epoch",
                    evaluation_strategy="epoch",
                    load_best_model_at_end=True
                ),
                train_dataset=self.prepare_dataset(),
                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            )
            
            print("Starting fine-tuning...")
            trainer.train()
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            raise

    def evaluate(self, test_data: List[Dict]) -> None:
        """Evaluate the model and save results."""
        results = {"responses": []}
        
        for example in tqdm(test_data, desc="Evaluating"):
            try:
                prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                response = self.tokenizer.decode(
                    self.model.generate(**inputs, max_length=200, temperature=0.7)[0],
                    skip_special_tokens=True
                )
                
                results["responses"].append({
                    "book": example["book"],
                    "context": example["context"],
                    "question": example["question"],
                    "expected": example["answer"],
                    "actual": response
                })
            except Exception as e:
                print(f"Error evaluating: {e}")
                continue

        # Save results
        df = pd.DataFrame(results["responses"])
        df.to_csv("harry_potter_evaluation_results.csv", index=False)
        print("\nEvaluation Summary by Book:")
        for book in df['book'].unique():
            print(f"\n{book}: {len(df[df['book'] == book])} examples")

def main():
    # Initialize the fine-tuner with Mistral model
    fine_tuner = HarryPotterFineTuner(model_name="mistralai/Mistral-7B-v0.1")

    # Load Harry Potter books
    books_dir = "path/to/your/harry_potter_books"
    fine_tuner.load_books(books_dir)

    # Fine-tune the model
    fine_tuner.fine_tune(output_dir="harry_potter_fine_tuned")

    # Split data into train and test sets
    train_data = fine_tuner.training_data[:int(len(fine_tuner.training_data) * 0.8)]
    test_data = fine_tuner.training_data[int(len(fine_tuner.training_data) * 0.8):]

    # Evaluate model on test data
    fine_tuner.evaluate(test_data)

if __name__ == "__main__":
    main() 