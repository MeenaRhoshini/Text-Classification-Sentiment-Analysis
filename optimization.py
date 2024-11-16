import torch
import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("reviews.csv")
df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))  # Text preprocessing

# Define features (X) and labels (y)
X = df['cleaned_text']
y = df['label']  # Assuming binary sentiment (0 = negative, 1 = positive)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()  # Vectorize the train data
X_test_vec = vectorizer.transform(X_test).toarray()  # Vectorize the test data

# Save the vectorizer for future use
joblib.dump(vectorizer, 'vectorizer.pkl')

# Now, you can pass X_train_vec to the optimization function
def optimize_model(X_train_vec, y_train):
    # Optimization code goes here
    print("Optimizing the model...")
    # Example: Print the shape of X_train_vec
    print(f"Shape of X_train_vec: {X_train_vec.shape}")

# Call the function with X_train_vec
optimize_model(X_train_vec, y_train)
