
import torch
import librosa
from torch.utils.data import Dataset

class VoiceCommandDataset(Dataset):
    def __init__(self, audio_encodings, text_encodings, labels):
        self.audio_encodings = audio_encodings
        self.text_encodings = text_encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.audio_encodings.items()}
        item.update({key: torch.tensor(val[idx]) for key, val in self.text_encodings.items()})
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_audio(tokenizer, audio_paths):
    audio_encodings = tokenizer(
        [librosa.load(audio_path, sr=16000)[0] for audio_path in audio_paths],
        return_tensors='pt',
        padding='longest'
    )
    return audio_encodings

def preprocess_text(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
