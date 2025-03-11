# Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset (Make sure the file path is correct)
data = pd.read_csv('spam.csv', encoding='latin-1')

# Check the first few rows of the dataset
print(data.head())

# Clean the dataset
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Check for missing data
print(data.isnull().sum())

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Define a function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply the preprocessing function to the messages
data['message'] = data['message'].apply(preprocess_text)

# Convert text data into numerical format using CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data (X)
X = vectorizer.fit_transform(data['message'])

# Labels (spam or ham)
y = data['label']

# Check the shape of the feature matrix
print(X.shape)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape[0]}")
print(f"Test data size: {X_test.shape[0]}")

# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

new_messages = [
    "Congrats! You've won a $1000 gift card. Click here to claim it!",
    "Hey, are you free for a coffee tomorrow?"
]

# Preprocess and vectorize the new messages
new_messages = [preprocess_text(msg) for msg in new_messages]
new_messages_vectorized = vectorizer.transform(new_messages)

# Predict using the trained model
predictions = model.predict(new_messages_vectorized)

# Display predictions
for msg, pred in zip(new_messages, predictions):
    print(f"Message: {msg}")
    print(f"Prediction: {'Spam' if pred == 'spam' else 'Ham'}")

# Check the cleaned data
print(data.head())
