from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize Flask application
app = Flask(__name__)

# Step 1: Load and Prepare the Dataset
# Load your dataset here
dataset_path = 'Reviews.csv'  # Replace with your local path
data = pd.read_csv(dataset_path)

def convert_to_sentiment(score):
    if score >= 4:
        return 'positive'
    elif score == 3:
        return 'neutral'
    else:
        return 'negative'

data['sentiment'] = data['Score'].apply(convert_to_sentiment)
data.dropna(subset=['Text'], inplace=True)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(data['Text'])
y = data['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Define a function to predict sentiment for new reviews
def predict_sentiment(review):
    review_transformed = tfidf.transform([review])
    sentiment_code = model.predict(review_transformed)[0]
    sentiment_label = {2: 'positive', 1: 'neutral', 0: 'negative'}
    return sentiment_label[sentiment_code]

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML file called 'index.html'

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input review from the form
    review = request.form.get('review')
    
    # Predict sentiment
    sentiment = predict_sentiment(review)
    
    # Return the result as a JSON response
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
