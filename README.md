# Intel Unnati GenAI Project

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project, developed for Intel Unnati, demonstrates the integration of generative AI and sentiment analysis using OpenVINO optimization. It combines a GPT-2 model for text generation with a DistilBERT model for sentiment analysis, showcasing how these technologies can work together to create sentiment-aware text generation.

## Features

- Text generation using GPT-2 model
- Sentiment analysis using DistilBERT model optimized with OpenVINO
- Interactive command-line interface for user input
- Sentiment-guided text generation
- Sentiment verification of generated text

## Prerequisites

- Python 3.7+
- pip package manager

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/DANICKA-SHARON/Intel-Internship_Running-GenAI-on-Intel-AI-Laptops-and-Simple-LLM-Inference/tree/main
   ```

2. Install the required packages:
   ```
   pip install openvino transformers datasets numpy tqdm torch
   ```

## Usage

Run the main script to start the interactive session:

```
python main.py
```

Follow the prompts to:
1. Enter the desired sentiment (Positive or Negative)
2. Provide the beginning of a statement
3. Receive generated text and its analyzed sentiment

Enter 'quit' when prompted for sentiment to exit the program.

## Project Structure

- `main.py`: The main script containing the `GenAIProject` class and the interactive loop
- `sentiment_model.onnx`: ONNX format of the sentiment analysis model
- `sentiment_model.xml`: OpenVINO IR format of the sentiment analysis model
- `README.md`: This file

## Technical Details

### Models Used
- Text Generation: GPT-2 (`gpt2`)
- Sentiment Analysis: DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)

### OpenVINO Integration
The sentiment analysis model is converted to OpenVINO Intermediate Representation (IR) format for optimized inference. This demonstrates the use of Intel's OpenVINO toolkit to enhance performance.

### Workflow
1. The GPT-2 model generates text based on user input and specified sentiment.
2. The OpenVINO-optimized DistilBERT model analyzes the sentiment of the generated text.
3. The program compares the intended sentiment with the analyzed sentiment for verification.

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

