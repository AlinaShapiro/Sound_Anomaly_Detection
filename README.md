# Sound Anomaly Detection 🎵🔍

A deep learning project for detecting anomalous sounds (e.g., footsteps, breaking glass, screams) in audio streams using transformer-based models.


## 📜 Project Overview

### Objective
Develop a robust system to identify unusual or dangerous sound events in real-world environments using state-of-the-art audio models.

**Target Anomalies**:
- Footsteps in restricted areas
- Breaking glass
- Human screams/shouts
- Other context-specific anomalies

## Core Components
### 📂 Dataset

**Kaggle Dataset**: [AudioAnomalyDataset](https://www.kaggle.com/datasets/ahmedabbasi/audioanomalydataset)  
- Contains labeled audio samples of:
  - Normal sounds (background noise, conversations)
  - Anomalous events (breaking glass, screams, footsteps)
- Format: `.wav` files at 16kHz sampling rate

### 🧠 Models

  - Transformer-based models like [**Wav2Vec2**](https://huggingface.co/facebook/wav2vec2-base-960h), [**Whisper**](https://huggingface.co/openai/whisper-base).

## Applications

This project can be applied to both **Security Systems** and **Smart Home Automation**, as sound anomaly detection can expand the capabilities of traditional security systems. These systems often rely on motion sensors or magnetic contacts to detect intruders, but sound anomaly detection allows them to detect sounds like broken glass, footsteps, or quiet speech, which may indicate an intruder. This is particularly useful for large spaces where conventional sensors may not be able to detect intruders effectively.
