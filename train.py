import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
import re
from model import create_model  # Make sure this is defined correctly in model.py

# Load the dataset (ensure you have the correct path to your CSV file)
df = pd.read_csv("reviews.csv")
df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))  # Example text preprocessing

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

# Convert to torch tensors
X_train_vec = torch.tensor(X_train_vec, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape to match output size

X_test_vec = torch.tensor(X_test_vec, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # Reshape to match output size

# Instantiate the model
model = create_model(input_dim=X_train_vec.shape[1])  # Use the number of features as input dimension
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(X_train, y_train, model, criterion, optimizer, epochs=5, batch_size=32):
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct_preds = 0
        total_preds = 0

        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = (outputs > 0.5).float()  # Binary classification
            correct_preds += (predictions == labels).sum().item()
            total_preds += labels.size(0)

        # Print training statistics
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {correct_preds / total_preds}")

    # Save the trained model weights after training
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

# Train the model
train_model(X_train_vec, y_train, model, criterion, optimizer)
