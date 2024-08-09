
import torch
import argparse
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, BertForSequenceClassification, BertTokenizer
from utils import get_device, preprocess_audio, preprocess_text, VoiceCommandDataset

def main(model_path, data_path):
    # Load Models and Tokenizers
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained(os.path.join(model_path, 'wav2vec'))
    wav2vec_tokenizer = Wav2Vec2Tokenizer.from_pretrained(os.path.join(model_path, 'wav2vec'))
    bert_model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, 'bert'))
    bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, 'bert'))

    # Device
    device = get_device()
    wav2vec_model.to(device)
    bert_model.to(device)

    # Load Dataset
    dataset = pd.read_csv(data_path)
    val_audio, val_texts, val_labels = dataset['audio'], dataset['text'], dataset['label']
    val_audio_encodings = preprocess_audio(wav2vec_tokenizer, val_audio.tolist())
    val_text_encodings = preprocess_text(bert_tokenizer, val_texts.tolist(), max_length=128)

    # DataLoader
    val_dataset = VoiceCommandDataset(val_audio_encodings, val_text_encodings, val_labels.tolist())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

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

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    # Evaluate
    avg_loss, accuracy = evaluate(wav2vec_model, bert_model, val_loader, device)
    print(f'Average Loss: {avg_loss}')
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing the fine-tuned models')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
