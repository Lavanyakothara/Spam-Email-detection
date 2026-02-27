import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'message': [
        'Win money now',
        'Hello how are you',
        'Claim your free prize',
        'Let us meet tomorrow',
        'Free lottery ticket',
        'Are you coming to class?',
        'Congratulations you won cash prize',
        'Important meeting tomorrow'
    ],
    'label': ['spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Custom message prediction
while True:
    msg = input("Enter a message (or type 'exit'): ")
    if msg.lower() == 'exit':
        break

    msg_vector = vectorizer.transform([msg])
    result = model.predict(msg_vector)

    print("Prediction:", result[0])