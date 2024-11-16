import torch
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from model import create_model
import re

# Load the model and vectorizer
model = create_model(input_dim=29)  # Set the input dimension to 5000 (same as during training)
model.load_state_dict(torch.load('model.pth'))
model.eval()

vectorizer = joblib.load('vectorizer.pkl')

# Load and preprocess test data
df_test = pd.read_csv("reviews.csv")  # Assuming you have the test data as well
df_test['cleaned_text'] = df_test['text'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))  # Text preprocessing

# Vectorize the test data
X_test = df_test['cleaned_text']
y_test = df_test['label']

X_test_vec = vectorizer.transform(X_test).toarray()  # Transform the test data using the loaded vectorizer
X_test_vec = torch.tensor(X_test_vec, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# Define the evaluation function
def evaluate_model(X_test, y_test, model):
    with torch.no_grad():  # Disable gradient calculation for evaluation
        model.eval()  # Set model to evaluation mode
        outputs = model(X_test)
        predictions = (outputs > 0.5).float()  # Binary classification

        # Calculate accuracy
        correct_preds = (predictions == y_test).sum().item()
        accuracy = correct_preds / y_test.size(0)

        print(f"Accuracy: {accuracy * 100:.2f}%")


# Evaluate the model
evaluate_model(X_test_vec, y_test, model)
