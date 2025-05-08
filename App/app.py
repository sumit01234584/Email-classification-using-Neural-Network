from flask import Flask, render_template_string, request
import torch
import pickle
import numpy as np
from model import SpamClassifier

app = Flask(__name__)

# Load vocab dictionary
with open("dataset/word_column_dict.pkl", "rb") as f:
    word_column_dict = pickle.load(f)

vocab_size = len(word_column_dict)

# Load model
model = SpamClassifier(vocab_size)
model.load_state_dict(torch.load("checkpoints/spam_classifier_2.pth", map_location=torch.device('cpu')))
model.eval()

# HTML Template
HTML_TEMPLATE = """
<!doctype html>
<title>Spam Classifier</title>
<h2>Spam or Ham Classifier</h2>
<form method=post>
  <textarea name=text cols=60 rows=6 placeholder="Enter your message here...">{{ text }}</textarea><br><br>
  <input type=submit value=Classify>
</form>
{% if prediction is not none %}
  <h3>Prediction: <span style="color:{{ color }}">{{ prediction }}</span></h3>
{% endif %}
"""

# Helper: Vectorize input
def vectorize_text(text, word_column_dict, vocab_size):
    vector = np.zeros(vocab_size)
    for word in text.split():
        if word in word_column_dict:
            vector[word_column_dict[word]] += 1
    return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

@app.route('/', methods=['GET', 'POST'])
def classify():
    prediction = None
    color = "black"
    text = ""

    if request.method == 'POST':
        text = request.form['text']
        input_tensor = vectorize_text(text, word_column_dict, vocab_size)
        with torch.no_grad():
            output = model(input_tensor)
            pred = (output.item() > 0.5)
            prediction = "SPAM" if pred else "HAM"
            color = "red" if pred else "green"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, color=color, text=text)

if __name__ == '__main__':
    app.run(debug=True)
