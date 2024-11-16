import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a simple neural network for sentiment analysis
class SentimentAnalysisModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SentimentAnalysisModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation for hidden layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, x):
        x = self.fc1(x)  # Pass input through the first layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Pass through the second layer
        x = self.sigmoid(x)  # Apply sigmoid for binary output
        return x

# Instantiate the model function
def create_model(input_dim):
    model = SentimentAnalysisModel(input_dim=input_dim)
    return model
