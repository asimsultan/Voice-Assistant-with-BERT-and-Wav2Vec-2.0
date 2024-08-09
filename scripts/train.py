
import os
import torch
import argparse
import pandas as pd
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, BertForSequenceClassification, BertTokenizer, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from utils import get_device, preprocess_audio, preprocess_text, VoiceCommandDataset

def main(data_path):
    # Parameters
    wav2vec_model_name = 'facebook/wav2vec2-base-960h'
    bert_model_name = 'bert-base-uncased'
    max_length = 128
    batch_size = 8
    epochs = 3
    learning_rate = 5e-5

    # Load Dataset
    dataset = pd.read_csv(data_path)
    train_audio, val_audio, train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset['audio'], dataset['text'], dataset['label'], test_size=0.1
    )

    # Tokenizers
    wav2vec_tokenizer = Wav2Vec2Tokenizer.from_pretrained(wav2vec_model_name)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Preprocess Data
    train_audio_encodings = preprocess_audio(wav2vec_tokenizer, train_audio.tolist())
    val_audio_encodings = preprocess_audio(wav2vec_tokenizer, val_audio.tolist())
    train_text_encodings = preprocess_text(bert_tokenizer, train_texts.tolist(), max_length)
    val_text_encodings = preprocess_text(bert_tokenizer, val_texts.tolist(), max_length)

    # DataLoader
    train_dataset = VoiceCommandDataset(train_audio_encodings, train_text_encodings, train_labels.tolist())
    val_dataset = VoiceCommandDataset(val_audio_encodings, val_text_encodings, val_labels.tolist())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Models
    device = get_device()
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_name)
    wav2vec_model.to(device)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=3)
    bert_model.to(device)

    # Optimizers and Schedulers
    wav2vec_optimizer = AdamW(wav2vec_model.parameters(), lr=learning_rate)
    bert_optimizer = AdamW(bert_model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    wav2vec_scheduler = get_scheduler(
        name="linear", optimizer=wav2vec_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    bert_scheduler = get_scheduler(
        name="linear", optimizer=bert_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(wav2vec_model, bert_model, data_loader, wav2vec_optimizer, bert_optimizer, device, wav2vec_scheduler, bert_scheduler):
        wav2vec_model.train()
        bert_model.train()
        total_loss = 0

        for batch in data_loader:
            audio_input_ids = batch['audio_input_ids'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Wav2Vec2 model forward pass
            wav2vec_outputs = wav2vec_model(audio_input_ids).logits

            # BERT model forward pass
            bert_outputs = bert_model(input_ids=text_input_ids, attention_mask=attention_mask, labels=labels)
            loss = bert_outputs.loss
            total_loss += loss.item()

            loss.backward()
            wav2vec_optimizer.step()
            wav2vec_scheduler.step()
            bert_optimizer.step()
            bert_scheduler.step()
            wav2vec_optimizer.zero_grad()
            bert_optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Evaluation Function
    def evaluate(wav2vec_model, bert_model, data_loader, device):
        wav2vec_model.eval()
        bert_model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                audio_input_ids = batch['audio_input_ids'].to(device)
                text_input_ids = batch['text_input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Wav2Vec2 model forward pass
                wav2vec_outputs = wav2vec_model(audio_input_ids).logits

                # BERT model forward pass
                bert_outputs = bert_model(input_ids=text_input_ids, attention_mask=attention_mask, labels=labels)
                loss = bert_outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(bert_outputs.logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(wav2vec_model, bert_model, train_loader, wav2vec_optimizer, bert_optimizer, device, wav2vec_scheduler, bert_scheduler)
        val_loss, val_accuracy = evaluate(wav2vec_model, bert_model, val_loader, device)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')
        print(f'Validation Loss: {val_loss}')
        print(f'Validation Accuracy: {val_accuracy}')

    # Save Models
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    wav2vec_model.save_pretrained(os.path.join(model_dir, 'wav2vec'))
    wav2vec_tokenizer.save_pretrained(os.path.join(model_dir, 'wav2vec'))
    bert_model.save_pretrained(os.path.join(model_dir, 'bert'))
    bert_tokenizer.save_pretrained(os.path.join(model_dir, 'bert'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing voice command data')
    args = parser.parse_args()
    main(args.data_path)
