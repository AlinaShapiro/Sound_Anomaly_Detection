import torch
import librosa
import numpy as np
import random
import argparse
from datasets import load_from_disk

from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification,
    WhisperProcessor, WhisperForConditionalGeneration
)
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import torchaudio.transforms as transforms

SEED = 42
DATASET_PATH = "./data/Audio-Anomaly-Dataset/Audio_dataset_splited"
CLASSES = ["foot", "police", "gunshoot", "scream", "glassbreak", "explosion", "babycry"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
AVAILABLE_MODELS = {
    "wav2vec2": "facebook/wav2vec2-base-960h",
    "whisper": "openai/whisper-base"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_processor(model_name, num_labels, device):
    if model_name == "wav2vec2":
        processor = Wav2Vec2Processor.from_pretrained(AVAILABLE_MODELS[model_name])
        model = Wav2Vec2ForSequenceClassification.from_pretrained(AVAILABLE_MODELS[model_name], num_labels=num_labels)
    elif model_name == "whisper":
        processor = WhisperProcessor.from_pretrained(AVAILABLE_MODELS[model_name])
        model = WhisperForConditionalGeneration.from_pretrained(AVAILABLE_MODELS[model_name])
    else:
        raise ValueError(f"Model name: {model_name}")
    return processor, model.to(device)

def audio_to_mel(audio, sr):
    mel_spec = transforms.MelSpectrogram(sample_rate=sr, n_mels=128)(torch.tensor(audio))
    mel_spec = librosa.power_to_db(mel_spec.numpy(), ref=np.max)
    return mel_spec

class AudioDataset(Dataset):
    def __init__(self, dataset_split, model_name):
        self.dataset = dataset_split
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_array = self.dataset[idx]["array"]
        label = CLASS_TO_IDX[self.dataset[idx]["label"]]
        return audio_array, label

def collate_fn_factory(model_name, processor, device):
    def collate_fn(batch):
        audios, labels = zip(*batch)
        inputs = processor(list(audios), sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.to(device), torch.tensor(labels, dtype=torch.long).to(device)
    return collate_fn

def evaluate(model, data_loader, model_name, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            if model_name in ["wav2vec2"]:
                outputs = model(inputs).logits
                preds = torch.argmax(outputs, dim=1)
            elif model_name in ["whisper"]:
                outputs = model.generate(inputs)
                preds = torch.zeros_like(labels) 
            else:
                raise ValueError(f"Unknown Model Name: {model_name}")
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    return classification_report(y_true, y_pred, target_names=CLASSES)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=AVAILABLE_MODELS.keys(), default="wav2vec2", help="Choosing a model for inference")
    args = parser.parse_args()
    model_name = args.model_name

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor, model = load_model_and_processor(model_name, len(CLASSES), device)

    dataset = load_from_disk(DATASET_PATH)
    test_dataset = AudioDataset(dataset["test"], model_name)
    collate_fn = collate_fn_factory(model_name, processor, device)

    g = torch.Generator()
    g.manual_seed(SEED)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, worker_init_fn=lambda _: np.random.seed(SEED), generator=g)

    report = evaluate(model, test_loader, model_name, device)
    print(report)

if __name__ == "__main__":
    main()
