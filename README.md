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
```

---

## Project Snippet

<p align="center">
  <img src="https://github.com/user-attachments/assets/89b27d92-6fe3-469a-83ca-6c89691e8a3b" height="250" /> <br>
  <img src="https://github.com/user-attachments/assets/caba38fc-634a-40fc-b076-45c2b77d7ed9" height="250" />
  <img src="https://github.com/user-attachments/assets/38cc3d72-826d-4384-abfb-9582f976d352" height="250" />
  <img src="https://github.com/user-attachments/assets/ace04df4-0823-4657-bd4d-08ad2bcea7ac" height="250" />
</p>

