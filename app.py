from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
mlp_model = load_model('mlp_model.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    preprocessed_input = tfidf_vectorizer.transform([input_text])
    
    preprocessed_input_dense = preprocessed_input.toarray()
    
    prediction = mlp_model.predict(preprocessed_input_dense)
    probability = prediction[0][0]  # Get the probability of the positive class

    result = {
        'prediction': int(probability >= 0.5),
        'probability': probability
    }

    # Set the depression level and message based on the probability
    depression_level = ""
    message = ""

    if probability < 0.1:
        depression_level = "No Depression"
        message = "No depression detected. Continue maintaining a healthy lifestyle. If you ever feel down or need support, don't hesitate to talk to someone you trust."
    elif 0.1 <= probability <= 0.4:
        depression_level = "Level 1: Mild"
        message = "It seems you might be experiencing mild symptoms of depression. It's a good idea to talk to a friend or family member about how you're feeling. Engaging in regular physical activity, maintaining a balanced diet, and practicing mindfulness can also help improve your mood."
    elif 0.41 <= probability <= 0.7:
        depression_level = "Level 2: Moderate"
        message = "It appears you may be experiencing moderate symptoms of depression. Consider seeking support from a mental health professional. Therapy or counseling can be very beneficial. Additionally, try to engage in activities that you enjoy and that help you relax."
    elif probability > 0.7:
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

    if input_text.strip() == "":
        result['depression_level'] = ""
        result['recommendation'] = "Please enter some text to analyze."

    probability_percentage = round(probability * 100, 0)
  
    return render_template('index.html',
                           depression_level=depression_level,
                           message=message,
                           progress_bar_color=progress_bar_color,
                           probability=probability_percentage,
                           input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
