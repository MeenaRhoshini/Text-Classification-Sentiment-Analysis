import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Clean and preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    words = word_tokenize(text)  # Tokenize text into words
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)  # Join words back into a string

# Load data
df = pd.read_csv("reviews.csv")  # Assuming you have a CSV with 'text' and 'label'
df['cleaned_text'] = df['text'].apply(preprocess_text)  # Apply preprocessing to each review

# Split into training and testing sets
X = df['cleaned_text']
y = df['label']  # Assuming binary sentiment (0 = negative, 1 = positive)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save the vectorizer for future use
joblib.dump(vectorizer, 'vectorizer.pkl')
