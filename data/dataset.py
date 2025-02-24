import os
import random
import librosa
import numpy as np
import pickle as pkl
from tqdm import tqdm 

from collections import defaultdict
from datasets import Dataset, DatasetDict

root_dir = "./data/Audio-Anomaly-Dataset/Audio_dataset"

def load_dataset(root_dir):
    data = defaultdict(list)
    
    for subset in os.listdir(root_dir):
        subset_path = os.path.join(root_dir, subset)
        if os.path.isdir(subset_path):
            for file_name in tqdm(os.listdir(subset_path), desc=f"Processing {subset}..."):
                if file_name.endswith(".wav"):
        
                    class_name = file_name.split("_")[0]
                    file_path = os.path.join(subset_path, file_name)
                    data["file"].append(file_path)
                    data["array"].append(librosa.load(file_path, sr=16000, mono=True)[0])
                    data["label"].append(class_name)
                    data["subset"].append(subset)
    
    return data

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    split_data = {"train": defaultdict(list), "validation": defaultdict(list), "test": defaultdict(list)}
    
    grouped_data = defaultdict(list)
    for i in range(len(data["file"])):
        key = (data["label"][i], data["subset"][i])
        grouped_data[key].append(i)
    
    for key, indices in grouped_data.items():
        random.shuffle(indices)
        total = len(indices)
        
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        
        for i, idx in enumerate(indices):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "validation"
            else:
                split = "test"
            
            for field in data:
                split_data[split][field].append(data[field][idx])
    
    return split_data

if __name__=="__main__":

    data = load_dataset(root_dir)

    split_data = split_dataset(data)

    train_dataset = Dataset.from_dict(split_data["train"])
    val_dataset = Dataset.from_dict(split_data["validation"])
    test_dataset = Dataset.from_dict(split_data["test"])

    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    # print(final_dataset['train'][0]['file'])
    # with open("final_dataset.pkl", "wb") as f:
    #     pkl.dump(final_dataset, f)

    # with open("final_dataset.pkl", "rb") as f:
    #     loaded_dataset = pkl.load(f)

    # final_dataset.save_to_disk("./data/Audio-Anomaly-Dataset/Audio_dataset_splited")

    print(final_dataset)