from flask import Flask, render_template, request
import pickle
import re
import nltk
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

xgb_model = pickle.load(open("xgb_model_bal.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(cleaned)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    user_prompt = ""

    if request.method == "POST":
        user_prompt = request.form["prompt"]

        if len(user_prompt.split()) <= 2:
            prediction = "Low"
        else:
            cleaned_prompt = preprocess_text(user_prompt)
            vector = tfidf.transform([cleaned_prompt])

            pred_num = xgb_model.predict(vector)[0]

            prediction = label_encoder.inverse_transform([pred_num])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        prompt=user_prompt
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



