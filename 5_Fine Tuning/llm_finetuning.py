from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

# Load environment variables
load_dotenv()

class TrainingExample(BaseModel):
    """Schema for training examples"""
    input_text: str = Field(description="The input text for the model")
    output_text: str = Field(description="The expected output text")
    context: Optional[str] = Field(description="Additional context if needed")

class LLMFineTuner:
    def __init__(self, api_key: str):
        """Initialize the fine-tuning process."""
        self.api_key = api_key
        self.llm = OpenAI(
            temperature=0.7,
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo"
        )
        self.training_data = []
        self.fine_tuned_model = None

    def load_training_data(self, directory_path: str) -> None:
        """Load and prepare training data from a directory."""
        try:
            # Load documents from directory
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)

            # Convert to training examples
            for doc in texts:
                # Here you would implement your logic to create input/output pairs
                # This is a simple example where we split the text into input and output
                text = doc.page_content
                if len(text) > 100:  # Only process texts that are long enough
                    input_text = text[:50]
                    output_text = text[50:100]
                    self.training_data.append(
                        TrainingExample(
                            input_text=input_text,
                            output_text=output_text,
                            context=doc.metadata.get("source", "")
                        )
                    )

            print(f"Loaded {len(self.training_data)} training examples")

        except Exception as e:
            print(f"Error loading training data: {e}")
            raise

    def prepare_fine_tuning_data(self, output_file: str) -> None:
        """Prepare data in the format required for fine-tuning."""
        try:
            # Convert training data to the format required by OpenAI
            fine_tuning_data = []
            for example in self.training_data:
                fine_tuning_data.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": example.input_text},
                        {"role": "assistant", "content": example.output_text}
                    ]
                })

            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(fine_tuning_data, f, indent=2)

            print(f"Prepared fine-tuning data saved to {output_file}")

        except Exception as e:
            print(f"Error preparing fine-tuning data: {e}")
            raise

    def evaluate_model(self, test_data: List[TrainingExample]) -> dict:
        """Evaluate the model on test data."""
        results = {
            "correct": 0,
            "total": len(test_data),
            "responses": []
        }

        for example in tqdm(test_data, desc="Evaluating model"):
            try:
                # Create a chain for evaluation
                prompt = PromptTemplate(
                    input_variables=["input_text"],
                    template="Input: {input_text}\nOutput:"
                )
                chain = LLMChain(llm=self.llm, prompt=prompt)

                # Get model response
                response = chain.run(input_text=example.input_text)
                
                # Compare with expected output (simple exact match for this example)
                if response.strip() == example.output_text.strip():
                    results["correct"] += 1

                results["responses"].append({
                    "input": example.input_text,
                    "expected": example.output_text,
                    "actual": response
                })

            except Exception as e:
                print(f"Error evaluating example: {e}")
                continue

        results["accuracy"] = results["correct"] / results["total"]
        return results

    def save_evaluation_results(self, results: dict, output_file: str) -> None:
        """Save evaluation results to a file."""
        try:
            # Convert to DataFrame for better visualization
            df = pd.DataFrame(results["responses"])
            df.to_csv(output_file, index=False)
            print(f"Evaluation results saved to {output_file}")

            # Print summary
            print("\nEvaluation Summary:")
            print(f"Total examples: {results['total']}")
            print(f"Correct predictions: {results['correct']}")
            print(f"Accuracy: {results['accuracy']:.2%}")

        except Exception as e:
            print(f"Error saving evaluation results: {e}")
            raise

def main():
    # Initialize the fine-tuner
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")

    fine_tuner = LLMFineTuner(api_key)

    # Load training data
    training_dir = "path/to/your/training/data"
    fine_tuner.load_training_data(training_dir)

    # Prepare fine-tuning data
    fine_tuner.prepare_fine_tuning_data("fine_tuning_data.json")

    # Split data into train and test sets (simple example)
    train_data = fine_tuner.training_data[:int(len(fine_tuner.training_data) * 0.8)]
    test_data = fine_tuner.training_data[int(len(fine_tuner.training_data) * 0.8):]

    # Evaluate model on test data
    results = fine_tuner.evaluate_model(test_data)

    # Save evaluation results
    fine_tuner.save_evaluation_results(results, "evaluation_results.csv")

if __name__ == "__main__":
    main() 