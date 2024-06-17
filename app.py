import re

import joblib
import nltk
import numpy as np
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and vectorizer
svm_model = joblib.load('svm_model.pkl')
mlp_model = load_model('mlp_model.h5')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert text to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]

    # Join tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def home_test():
    return render_template('test.html')

@app.route('/predict-svm', methods=['POST'])
def predict_svm():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_test.pkl')
    input_text = request.form['input_text']
    preprocessed_input = tfidf_vectorizer.transform([input_text])
    
    preprocessed_input_dense = preprocessed_input.toarray()
    
    prediction = svm_model.predict(preprocessed_input_dense)
    probability = svm_model.predict_proba(preprocessed_input_dense)
    result = {
        'prediction': int(prediction[0]),
        'probability': probability[0].tolist()
    }
    # Set the depression level and message based on the probability
    depression_level = ""
    message = ""
    # if prediction[0] == 1:
    prob = probability[0][1]
    if prob < 0.1:
        depression_level = "No Depression"
        message = "No depression detected. Continue maintaining a healthy lifestyle. If you ever feel down or need support, don't hesitate to talk to someone you trust."
    elif 0.1 < prob < 0.4:
        depression_level = "Level 1: Mild"
        message = "It seems you might be experiencing mild symptoms of depression. It's a good idea to talk to a friend or family member about how you're feeling. Engaging in regular physical activity, maintaining a balanced diet, and practicing mindfulness can also help improve your mood."
    elif 0.41 < prob < 0.7:
        depression_level = "Level 2: Moderate"
        message = "It appears you may be experiencing moderate symptoms of depression. Consider seeking support from a mental health professional. Therapy or counseling can be very beneficial. Additionally, try to engage in activities that you enjoy and that help you relax."
    elif prob >= 0.7:
        depression_level = "Level 3: Severe"
        message = "Your responses indicate that you may be experiencing severe symptoms of depression. It is highly recommended to seek immediate help from a mental health professional. Don't hesitate to reach out to friends or family members for support as well."

    if depression_level == "No Depression":
        progress_bar_color = "bg-success"
    elif depression_level == "Level 1: Mild":
        progress_bar_color = "bg-success"
    elif depression_level == "Level 2: Moderate":
        progress_bar_color = "bg-warning"
    else:
        progress_bar_color = "bg-danger"


    result['depression_level'] = depression_level
    result['recommendation'] = message

    if(input_text.strip() == ""):
        result['depression_level'] = ""
        result['recommendation'] = "Please enter some text to analyze."

    probability = round(probability[0][1]*100, 0)
    # return jsonify(result)
    return render_template('test.html',depression_level= depression_level, message= message,progress_bar_color=progress_bar_color,probability=probability, input_text= input_text)


@app.route('/predict-mlp', methods=['POST'])
def predict():
    input_text = request.form['input_text']

    # Preprocess the input text
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    preprocessed_input = preprocess_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([preprocessed_input]).toarray()

    # Perform prediction
    prediction_prob = mlp_model.predict(input_vectorized)[0][0]
    # prediction = int(prediction_prob >= 0.5)

    # Determine depression level and recommendation
    if prediction_prob < 0.1:
        depression_level = "No Depression"
        recommendation = "No depression detected. Continue maintaining a healthy lifestyle. If you ever feel down or need support, don't hesitate to talk to someone you trust."
        progress_bar_color = "bg-success"

    elif prediction_prob < 0.35:
        depression_level = "Level 1: Mild"
        recommendation = "It seems you might be experiencing mild symptoms of depression. It's a good idea to talk to a friend or family member about how you're feeling. Engaging in regular physical activity, maintaining a balanced diet, and practicing mindfulness can also help improve your mood."
        progress_bar_color = "bg-success"

    elif 0.5 < prediction_prob < 0.51 :
        depression_level = "Please enter a valid statement"
        recommendation = ""
        progress_bar_color = "bg-secondary"
        prediction_prob = 0

    elif prediction_prob < 0.7:
        depression_level = "Level 2: Moderate"
        recommendation = "It appears you may be experiencing moderate symptoms of depression. Consider seeking support from a mental health professional. Therapy or counseling can be very beneficial. Additionally, try to engage in activities that you enjoy and that help you relax."
        progress_bar_color = "bg-warning"

    else:
        depression_level = "Level 3: Severe"
        recommendation = "Your responses indicate that you may be experiencing severe symptoms of depression. It is highly recommended to seek immediate help from a mental health professional. Don't hesitate to reach out to friends or family members for support as well."
        progress_bar_color = "bg-danger"

    probability_percentage = round(prediction_prob*100, 2)

    return render_template('index.html',
                           input_text=input_text,
                           depression_level=depression_level,
                           recommendation=recommendation,
                           progress_bar_color=progress_bar_color,
                           probability=probability_percentage)

if __name__ == '__main__':
    app.run(debug=True)
