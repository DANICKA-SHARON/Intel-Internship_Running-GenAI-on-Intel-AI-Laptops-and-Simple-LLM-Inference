# Import required libraries
import os
import openvino as ov  # OpenVINO library for optimized inference
import numpy as np     # NumPy for numerical operations
import torch           # PyTorch for deep learning operations
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)  # Hugging Face Transformers for pre-trained models

class GenAIProject:
    def __init__(self, gpt_model_name="gpt2", sentiment_model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the GenAIProject with specified models.
        
        :param gpt_model_name: Name of the GPT model to use for text generation (default: "gpt2")
        :param sentiment_model_name: Name of the sentiment analysis model (default: "distilbert-base-uncased-finetuned-sst-2-english")
        """
        # Store model names
        self.gpt_model_name = gpt_model_name
        self.sentiment_model_name = sentiment_model_name
        
        # Initialize model and tokenizer attributes as None
        self.gpt_tokenizer = None
        self.gpt_model = None
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.compiled_sentiment_model = None

    def setup_models(self):
        """
        Set up the GPT and sentiment analysis models.
        This method loads the pre-trained models and tokenizers, and prepares them for use.
        
        :return: True if setup is successful, False otherwise
        """
        try:
            # Set up GPT model
            # Load the tokenizer and model for text generation
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
            self.gpt_model = AutoModelForCausalLM.from_pretrained(self.gpt_model_name)
            
            # Ensure the tokenizer has a padding token
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
                self.gpt_model.config.pad_token_id = self.gpt_model.config.eos_token_id
            
            print(f"GPT model {self.gpt_model_name} loaded successfully.")

            # Set up sentiment model
            # Load the tokenizer and model for sentiment analysis
            self.sentiment_tokenizer = DistilBertTokenizer.from_pretrained(self.sentiment_model_name)
            self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(self.sentiment_model_name)
            print(f"Sentiment model {self.sentiment_model_name} loaded successfully.")

            # Convert sentiment model to OpenVINO for optimized inference
            self.convert_sentiment_model_to_openvino()
            
            return True
        except Exception as e:
            print(f"Error setting up models: {e}")
            return False

    def convert_sentiment_model_to_openvino(self):
        """
        Convert the sentiment analysis model to OpenVINO format for optimized inference.
        This method exports the PyTorch model to ONNX, then converts it to OpenVINO IR format.
        """
        try:
            # Create a dummy input for ONNX export
            dummy_input = self.sentiment_tokenizer("Hello, how are you?", return_tensors="pt")
            
            # Export the PyTorch model to ONNX format
            torch.onnx.export(self.sentiment_model, 
                              (dummy_input['input_ids'], dummy_input['attention_mask']),
                              "sentiment_model.onnx",
                              input_names=['input_ids', 'attention_mask'],
                              output_names=['output'],
                              dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                            'attention_mask': {0: 'batch_size', 1: 'sequence'},
                                            'output': {0: 'batch_size'}},
                              opset_version=11)
            
            # Convert ONNX model to OpenVINO IR format
            ov_model = ov.convert_model("sentiment_model.onnx")
            ov.save_model(ov_model, "sentiment_model.xml")
            
            # Compile the OpenVINO model for CPU execution
            core = ov.Core()
            model = core.read_model("sentiment_model.xml")
            self.compiled_sentiment_model = core.compile_model(model, "CPU")
            
            print("Sentiment model converted to OpenVINO IR format and compiled.")
        except Exception as e:
            print(f"Error converting sentiment model to OpenVINO: {e}")

    def generate_text(self, input_text, sentiment):
        """
        Generate text based on input and desired sentiment using the GPT model.
        
        :param input_text: The beginning of the statement to generate from
        :param sentiment: Desired sentiment (Positive or Negative)
        :return: Generated text or None if generation fails
        """
        try:
            # Dictionary of sentiment-specific prompts to guide text generation
            sentiment_prompts = {
                "Positive": [
                    "This wonderful", "The amazing", "Incredibly, this", "Fortunately,", "To everyone's delight,",
                    "Happily,", "Excitingly,", "Joyfully,", "Miraculously,", "In a stroke of good luck,"
                ],
                "Negative": [
                    "This terrible", "The awful", "Unfortunately, this", "Sadly,", "To everyone's dismay,",
                    "Regrettably,", "Disappointingly,", "Miserably,", "Tragically,", "In an unfortunate turn of events,"
                ]
            }

            # Select a random sentiment-specific prompt
            sentiment_words = sentiment_prompts[sentiment]
            prompt = f"{input_text} {np.random.choice(sentiment_words)}"
            
            # Tokenize the input for the GPT model
            inputs = self.gpt_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            # Generate text using the GPT model
            with torch.no_grad():
                outputs = self.gpt_model.generate(**inputs, max_length=100, num_return_sequences=1, temperature=0.7)
            
            # Decode the generated text
            generated_text = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the first sentence (up to the first period)
            first_sentence = generated_text.split('.')[0] + '.'
            return first_sentence
        except Exception as e:
            print(f"Error during text generation: {e}")
            return None

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text using the OpenVINO-optimized model.
        
        :param text: Text to analyze
        :return: "Positive" or "Negative", or None if analysis fails
        """
        try:
            # Tokenize the input text
            inputs = self.sentiment_tokenizer(text, return_tensors="np")
            
            # Perform inference using the OpenVINO-compiled model
            output = self.compiled_sentiment_model([inputs['input_ids'], inputs['attention_mask']])
            scores = output[self.compiled_sentiment_model.output(0)]
            
            # Determine the sentiment based on the output scores
            predicted_class = np.argmax(scores)
            return "Positive" if predicted_class == 1 else "Negative"
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return None

def main():
    """
    Main function to run the GenAI project interactively.
    This function sets up the project, and then enters a loop to generate
    and analyze text based on user input.
    """
    # Initialize the GenAIProject
    project = GenAIProject()
    
    # Set up the models
    if not project.setup_models():
        return  # Exit if model setup fails

    # Main interaction loop
    while True:
        # Get desired sentiment from user
        sentiment = input("Enter the desired sentiment (Positive/Negative) or 'quit' to exit: ").capitalize()
        if sentiment.lower() == 'quit':
            break
        if sentiment not in ['Positive', 'Negative']:
            print("Invalid sentiment. Please enter 'Positive' or 'Negative'.")
            continue

        # Get input text from user
        input_text = input("Enter the beginning of the statement: ")
        
        # Generate text based on input and sentiment
        generated_text = project.generate_text(input_text, sentiment)
        if generated_text:
            print(f"\nGenerated text: {generated_text}")
            
            # Analyze the sentiment of the generated text
            analyzed_sentiment = project.analyze_sentiment(generated_text)
            print(f"Analyzed sentiment: {analyzed_sentiment}")
            
            # Check if the analyzed sentiment matches the requested sentiment
            if analyzed_sentiment != sentiment:
                print("Note: The generated text's sentiment doesn't match the requested sentiment.")
        else:
            print("Text generation failed.")

# Entry point of the script
if __name__ == "__main__":
    main()
