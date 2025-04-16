import gradio as gr
import transformers
import peft
import torch
import bitsandbytes as bnb
from typing import Dict

class HarryPotterModel:
    def __init__(self, model_path: str):
        """Initialize the model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load base model and tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
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
        
        # Load fine-tuned weights
        self.model = peft.PeftModel.from_pretrained(
            self.model,
            model_path,
            is_trainable=False
        )
        self.model.eval()

    def generate_response(self, context: str, question: str) -> str:
        """Generate a response for the given context and question."""
        try:
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the answer part
            response = response.split("Answer:")[-1].strip()
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def create_gradio_interface(model_path: str):
    """Create and launch the Gradio interface."""
    model = HarryPotterModel(model_path)
    
    def process_input(context: str, question: str) -> str:
        if not context or not question:
            return "Please provide both context and question."
        return model.generate_response(context, question)
    
    # Create the Gradio interface
    iface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Textbox(
                lines=4,
                label="Context",
                placeholder="Enter the Harry Potter context here..."
            ),
            gr.Textbox(
                lines=2,
                label="Question",
                placeholder="Enter your question about the context..."
            )
        ],
        outputs=gr.Textbox(
            lines=4,
            label="Answer",
            placeholder="The model's answer will appear here..."
        ),
        title="Harry Potter Q&A Model",
        description="Ask questions about Harry Potter! Enter a context from the books and ask a question about it.",
        examples=[
            [
                "Harry Potter was a boy who lived with his aunt and uncle, the Dursleys, who treated him terribly. He slept in a cupboard under the stairs and was often bullied by his cousin Dudley.",
                "How did Harry Potter live with the Dursleys?"
            ],
            [
                "Hermione Granger was known for her exceptional magical abilities and dedication to her studies. She often helped Harry and Ron with their homework and was particularly skilled at casting spells.",
                "What were Hermione's notable characteristics?"
            ]
        ]
    )
    
    return iface

def main():
    # Path to your fine-tuned model
    model_path = "harry_potter_fine_tuned"
    
    # Create and launch the interface
    iface = create_gradio_interface(model_path)
    
    # Launch the interface
    iface.launch(
        share=True,  # Enable sharing to get a public URL
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860  # Default Gradio port
    )

if __name__ == "__main__":
    main() 