## Prompt Grader

This project is a Machine Learning–based web application that classifies user prompts into Low, Medium, or High quality.
It is built using Logistic Regression with SMOTE, TF‑IDF, and deployed using Flask.

The goal of the project is to evaluate prompt quality fairly, avoiding bias toward majority classes and ensuring meaningful predictions.

---

## Final Model
**XGBoost with SMOTE**

The final model was selected to handle class imbalance and provide more balanced predictions across all prompt quality classes.

---

## Tech Stack
- Python  
- Flask  
- TF-IDF  
- XGBoost  
- SMOTE  
- NLTK  

---

## How It Works
1. User enters a prompt  
2. Text is cleaned and vectorized using TF-IDF  
3. XGBoost model predicts the prompt quality  
4. Very short prompts are classified as **Low** using a rule-based check  

---

## Run the Application
```bash
pip install flask scikit-learn imbalanced-learn nltk xgboost
python app.py
