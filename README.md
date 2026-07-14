# 🧠 Anxiety Prediction System

A Machine Learning project that predicts an individual's anxiety level based on lifestyle, health, and demographic factors. The model uses **XGBoost Regressor** with **GridSearchCV** hyperparameter tuning to improve prediction accuracy and generalization.

## 📌 Project Overview

Mental health is influenced by various factors such as sleep patterns, stress levels, physical activity, caffeine intake, and family history. This project analyzes these factors and predicts an anxiety score using machine learning techniques.

The application includes data preprocessing, model training, evaluation, and an interactive Streamlit web interface for real-time predictions.

---

## 🚀 Features

* Data Cleaning and Preprocessing
* Label Encoding for Categorical Features
* Feature Scaling using StandardScaler
* XGBoost Regression Model
* Hyperparameter Optimization using GridSearchCV
* Model Performance Evaluation
* Interactive Streamlit Web Application
* Real-Time Anxiety Level Prediction

---

## 📊 Dataset Features

The model is trained using factors such as:

* Age
* Gender
* Occupation
* Sleep Hours
* Physical Activity
* Caffeine Intake
* Alcohol Consumption
* Smoking
* Family History of Anxiety
* Stress Level
* Heart Rate
* Diet Quality
* Screen Time
* And other health-related attributes

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib
* Matplotlib
* Seaborn

---

## 📈 Model Training

The model was optimized using **GridSearchCV** to find the best combination of hyperparameters.

Example parameters tuned:

* n_estimators
* learning_rate
* max_depth
* min_child_weight

This helped improve prediction performance and reduce overfitting.

---

## 📋 Project Workflow

1. Load Dataset
2. Data Cleaning
3. Encode Categorical Variables
4. Scale Features
5. Split Data into Training and Testing Sets
6. Train XGBoost Regressor
7. Hyperparameter Tuning with GridSearchCV
8. Evaluate Model Performance
9. Save Model and Scalers
10. Deploy with Streamlit

---

## 📊 Evaluation Metrics

The model performance can be evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/your-username/anxiety-prediction.git
cd anxiety-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Anxiety-Prediction/
│
├── app.py
├── anxiety_model.pkl
├── scaler.pkl
├── requirements.txt
├── dataset.csv
├── notebooks/
├── images/
└── README.md
```

---

## 🎯 Future Improvements

* Deep Learning Models (ANN)
* Explainable AI using SHAP
* Model Deployment on Cloud
* Enhanced Visualization Dashboard
* Multi-Class Anxiety Classification

---

## 👨‍💻 Author

Harsh Jariwala

Bachelor of Computer Applications (BCA)
Data Science Enthusiast | Machine Learning Learner

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
