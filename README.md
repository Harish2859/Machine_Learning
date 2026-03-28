# Titanic Survival Predictor: A Supervised Learning Implementation

### 🚢 Project Overview
This repository contains my implementation of a classification pipeline using the classic Titanic dataset. This project is part of my **Month 5: Supervised Learning** curriculum, focused on mastering algorithms that learn from labeled data.

The goal was not just to achieve high accuracy, but to understand the end-to-end Machine Learning workflow: from raw data cleaning to model interpretation.

### 🧠 Learning Objectives
* **Data Preprocessing:** Handling missing values (Imputation) and converting categorical text into numerical data using One-Hot Encoding.
* **Algorithm Comparison:** Implementing and comparing three different classification models:
  1. **Logistic Regression:** Understanding the Sigmoid function and decision boundaries.
  2. **Decision Trees:** Visualizing logic-based splitting and managing tree depth.
  3. **Random Forests:** Learning the power of Ensemble methods (bagging) to reduce overfitting.
* **Model Evaluation:** Using Accuracy scores and Confusion Matrices to interpret results.

### 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Google Colab

### 📈 Results
After testing multiple models, the **Random Forest** emerged as the strongest predictor:
* **Logistic Regression:** ~80% Accuracy
* **Decision Tree:** ~81% Accuracy
* **Random Forest:** ~82% Accuracy

### 🔍 Key Insight
By analyzing feature importance, the model confirmed the historical "women and children first" protocol, showing that `Sex` and `Pclass` were the most significant predictors of survival.

---
*Note: This is a learning implementation created as part of my data science journey. I am currently exploring how different hyperparameters affect model generalization.*

<img width="1668" height="844" alt="image" src="https://github.com/user-attachments/assets/2de7ac31-a915-470a-bd20-605b7871f6af" />

<img width="635" height="500" alt="image" src="https://github.com/user-attachments/assets/dc8fe3a1-113c-4e1a-8c8c-78827f60226a" />

<img width="890" height="706" alt="image" src="https://github.com/user-attachments/assets/637dc94c-2dca-4586-8c08-7e0d44a25c5b" />

Diabetes Clinical Optimizer (Week 3)An exploration into deep learning training dynamics, focusing on Optimization and Regularization using the Pima Indians Diabetes dataset.🔬 Project OverviewThe goal of this "Clinical Trial" was to build a neural network that doesn't just memorize patient data (overfitting) but actually learns to generalize and predict diabetes in unseen patients.🛠️ Key Concepts AppliedAdam Optimizer: Replaced standard Stochastic Gradient Descent (SGD) to benefit from adaptive learning rates and momentum.Mini-Batch Training: Implemented a DataLoader with a batch size of 32, resulting in 20 iterations per epoch. This balanced training speed with gradient stability.Dropout (Regularization): Added a Dropout layer ($p=0.5$) to prevent neuron co-dependency and combat the overfitting observed in early trials.📊 The "Clinical Trial" ResultsDuring development, I compared two distinct training phases:The Overfitting Phase: Without sufficient regularization, the model showed a classic "divergence"—training loss continued to drop while validation loss skyrocketed after Epoch 23.The Optimized Phase: By increasing Dropout and utilizing Adam’s efficiency, the validation curve stabilized. The model stopped "memorizing" and started "learning."Conclusion: The final model maintains a consistent validation loss, proving that the regularization techniques successfully forced the network to learn general biological patterns rather than specific training samples.

<img width="561" height="565" alt="image" src="https://github.com/user-attachments/assets/a93f94a5-3df4-4618-ae96-40ed9749712c" />


# Breast Cancer Classification (PyTorch)

This project implements a Deep Learning model using **PyTorch** to classify breast cancer tumors as Malignant or Benign based on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## 🚀 Features
* **GPU Acceleration**: Automatically detects and uses CUDA if available.
* **Data Preprocessing**: Handles missing values and performs feature scaling using `StandardScaler`.
* **Deep Learning Architecture**: A 3-layer fully connected Neural Network with ReLU activation and Dropout for regularization.
* **High Accuracy**: Achieves **~99% accuracy** on the test set.

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Framework**: PyTorch
* **Data**: Pandas, Scikit-Learn

## 📋 How to Run
1. **Install Dependencies**:
   ```bash
   pip install torch pandas scikit-learn



## RAG Learning Notebook (ragLearn.ipynb)
This notebook is included in the repo as **ragLearn.ipynb**. It was created in Google Colab and walks through ChromaDB setup, custom embeddings, vector search, metadata filters, and multimodal examples. The screenshots below are captured from that notebook.

## Custom Embedding Functions
<img width="1214" height="635" alt="image" src="https://github.com/user-attachments/assets/aa7c25f6-c592-4eb2-b224-1bfd2f012092" />

## Vector Search
<img width="1179" height="415" alt="image" src="https://github.com/user-attachments/assets/ca5bb405-0fc5-4fb9-b028-a9d1271af9dc" />

Full Text Search & Regex Search
<img width="1145" height="235" alt="image" src="https://github.com/user-attachments/assets/ca3dd41b-32f7-4f8b-b6fc-9fb1fdcc40de" />

Metadata Filtering
<img width="1507" height="223" alt="image" src="https://github.com/user-attachments/assets/daf27a74-5b22-4f2d-bf51-519c594c6b4b" />

Multimodal Embeddings

<img width="775" height="236" alt="image" src="https://github.com/user-attachments/assets/bba26085-711c-4bac-b29a-bf93bd7d3929" />



