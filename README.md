# heart-disease-ai-pytorch
End-to-end PyTorch pipeline trained on Kaggle
# Heart Disease Prediction AI 🫀

This project demonstrates an end-to-end Machine Learning pipeline using PyTorch.

## 🚀 Overview
1.  **Training**: Trained a Neural Network on Kaggle using the Heart Disease dataset.
2.  **Model**: A custom sequential Neural Network (PyTorch).
3.  **Deployment**: The model weights (`.pth`) were downloaded and integrated into a local Python script for inference.

## 🛠️ Tech Stack
* **Python**
* **PyTorch** (Neural Network)
* **Pandas & Scikit-Learn** (Preprocessing)

## 💻 How to Run
1.  Install dependencies:
    ```bash
    pip install torch numpy
    ```
2.  Run the prediction script:
    ```bash
    python predict.py
    ```

## 📊 Results
The model analyzes patient data (age, BP, cholesterol, etc.) and outputs a risk score.
