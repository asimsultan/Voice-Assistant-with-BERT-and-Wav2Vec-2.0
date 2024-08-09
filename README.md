
# Voice Assistant with BERT and Wav2Vec 2.0

Welcome to the Voice Assistant with BERT and Wav2Vec 2.0 project! This project focuses on building a voice assistant using the BERT and Wav2Vec 2.0 models.

## Introduction

Voice assistants can process and understand voice commands. In this project, we leverage BERT for natural language understanding and Wav2Vec 2.0 for automatic speech recognition using a dataset of voice commands and their corresponding text labels.

## Dataset

For this project, we will use a custom dataset of audio files and their corresponding text commands. You can create your own dataset and place it in the `data/voice_command_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Flask
- Librosa
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/voice_assistant_bert_wav2vec.git
cd voice_assistant_bert_wav2vec

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes audio files and their corresponding text commands. Place these files in the data/ directory.
# The data should be in a CSV file with three columns: audio, text, and label.

# To fine-tune the Wav2Vec 2.0 and BERT models for voice assistant tasks, run the following command:
python scripts/train.py --data_path data/voice_command_data.csv

# To evaluate the performance of the fine-tuned models, run:
python scripts/evaluate.py --model_path models/ --data_path data/voice_command_data.csv
