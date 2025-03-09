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

## Installation and Run

### Prerequisites

Before running the project, ensure you have the following installed:

* Python 3.10 or higher
* CUDA and cuDNN (if using GPU)
* Git
* Conda (Optional)

### 1. Clone the Repository
Clone the project repository to your local machine:

```bash
git clone https://github.com/AlinaShapiro/Sound_Anomaly_Detection.git
cd Sound_Anomaly_Detection
```
### 2. Set Up a Virtual Environment

```bash
conda create -n sound_anomaly_detection python=3.10
conda activate sound_anomaly_detection
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run

#### Inference

To run inference on a test subset of Audio-Anomaly-Dataset, use:

```bash
python inference.py --model_name wav2vec2
```

## Applications

This project can be applied to both **Security Systems** and **Smart Home Automation**, as sound anomaly detection can expand the capabilities of traditional security systems. These systems often rely on motion sensors or magnetic contacts to detect intruders, but sound anomaly detection allows them to detect sounds like broken glass, footsteps, or quiet speech, which may indicate an intruder. This is particularly useful for large spaces where conventional sensors may not be able to detect intruders effectively.
