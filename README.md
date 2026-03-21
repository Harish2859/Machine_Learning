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


