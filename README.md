# 🩺 Diabetes Prediction Using Machine Learning

This project builds a machine-learning pipeline to predict the likelihood of diabetes based on key medical attributes.  
It covers the complete workflow — from data loading and exploratory data analysis (EDA) to model training, evaluation, and hyperparameter tuning.

The project uses the **PIMA Diabetes Dataset** and implements a **K-Nearest Neighbors (KNN)** classifier along with performance evaluation techniques such as the **Confusion Matrix** and **ROC-AUC Curve**.

---

## 📌 Objectives

- Analyze relationships between health features and diabetes outcome  
- Perform data preprocessing and feature scaling  
- Build and evaluate a predictive ML model  
- Optimize model performance using **GridSearchCV**

---

## 📊 Dataset Features

The dataset includes the following attributes:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  
- Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 🧠 Project Workflow

1. **Importing Libraries**  
2. **Reading and Inspecting the Dataset**  
3. **Exploratory Data Analysis (EDA)**  
   - Distribution analysis  
   - Outlier inspection  
4. **Data Visualization**  
   - Feature trends and relationships  
5. **Correlation Analysis**  
6. **Feature Scaling (Standardization)**  
7. **Train–Test Split**  
8. **Model Building – K-Nearest Neighbors (KNN)**  
9. **Model Evaluation**  
   - Accuracy  
   - Confusion Matrix  
   - ROC-AUC Curve  
10. **Hyperparameter Tuning using GridSearchCV**

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn

---

## 🏆 Model Evaluation

The model performance is evaluated using:

- Accuracy Score  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC-AUC Curve  

👉 *(You can update this section with your actual scores and best parameters.)*

---

## ▶️ How to Run the Project

1️⃣ Install dependencies  
```bash
pip install -r requirements.txt
