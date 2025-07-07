#  Sign Language Recognition with ResNet50 + LSTM on LSA64 Dataset

This project builds a **deep learning pipeline** to recognize **Argentinian Sign Language (LSA)** signs from video, using:
- Transfer learning (**ResNet50**) to extract spatial features from frames
- Temporal modeling (**LSTM**) to learn motion & sequence patterns
- PyTorch for the complete implementation

> This repository shows the real end-to-end workflow: data loading, preprocessing, model architecture, training, and evaluation.


## **Project Goal**
Recognize sign language videos from the **LSA64 dataset** by:
- Extracting key frames from each video
- Using ResNet50 (pre-trained on ImageNet) to get per-frame features
- Feeding frame features into an LSTM network to learn temporal dynamics
- Training & evaluating the model on real data

Used the **LSA64 dataset** (64 signs, 10 subjects, 3,200 videos):  
🔗 [LSA64 dataset link](https://facundoq.github.io/datasets/lsa64/)

## **Project Structure**
```plaintext
sign-language-recognition-lsa64/
├── notebooks/
│   └── SLD.ipynb                # Main notebook with actual code & experiments
├── src/                         # 🛠Reusable modules matching notebook logic
│   ├── dataset.py               # PyTorch Dataset: video loading & frame extraction
│   ├── model.py                 # ResNet50 + LSTM model
│   ├── train.py                 # Training & evaluation loops
│   └── utils.py                 # Plotting etc.
├── requirements.txt
├── README.md
└── .gitignore



