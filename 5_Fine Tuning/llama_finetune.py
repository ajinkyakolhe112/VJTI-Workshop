import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipeline("Hey how are you doing today?")

#%%
import re

def clean_text(text):
    # Remove chapter headers (assuming they have a common pattern)
    text = re.sub(r'CHAPTER [IVX]+\s*', '', text)
    # Remove page numbers or other patterns if needed
    text = re.sub(r'\d+', '', text)  # Remove all numbers if they are not useful.
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

with open('harry_potter.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

cleaned_text = clean_text(raw_text)
print(cleaned_text[:1000]) # Print the first 1000 to see if everything is working properly
# Save the cleaned text
with open('harry_potter_cleaned.txt', 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

#%%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token # For padding, to make sure input sequence are in same length.

# Split the dataset into chunks of sequence length
def tokenize_function(examples, sequence_length = 512):
    chunk_size = sequence_length
    tokenized_text = tokenizer(examples["text"]) # tokenizer the whole text
    all_tokens = []
    for i in range(0, len(tokenized_text['input_ids']) - chunk_size, chunk_size):
        input_ids = tokenized_text['input_ids'][i : i + chunk_size]
        all_tokens.append(
            {
                "input_ids": input_ids,
                "attention_mask": tokenized_text['attention_mask'][i:i+chunk_size]
            }
        )
    return all_tokens


with open("harry_potter_cleaned.txt", 'r', encoding="utf-8") as f:
    text = f.read()

from datasets import Dataset
dataset = Dataset.from_dict({"text": [text]}) # using dataset to create dataset objects

tokenized_dataset = dataset.map(tokenize_function, batched = True, batch_size = 1) # batch size of 1 because the length of our text is one
print(tokenized_dataset)

#%%
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                device_map = "auto",
                                load_in_8bit = True
                                ) # Make sure model is loaded to the device

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./harry_potter_model",  # Where to save the fine-tuned model
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=1,   # Adjust as needed
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps = 50,
    #gradient_accumulation_steps = 8, # Enable if GPU memory is not sufficient
    learning_rate=2e-5,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    )

# Start training
trainer.train()

#%%
from transformers import pipeline
generator = pipeline('text-generation', model= "./harry_potter_model", tokenizer= tokenizer)

prompt = "Harry Potter was walking down the stairs and saw" # initial prompt
generated_text = generator(prompt, max_length=200, num_return_sequences = 1)[0]['generated_text']
print(generated_text)