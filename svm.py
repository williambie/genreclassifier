import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the data from the json file
with open('movies.json', 'r') as file:
    movies = json.load(file)

# Preprocess the movie data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Preprocess the movie plot and genre names and remove the title
for movie in movies:
    movie['description'] = preprocess_text(movie['description'])
    movie['genre_names'] = [genre.lower() for genre in movie['genre_names']]
    movie['genre'] = movie['genre_names'][0] if movie['genre_names'] else 'unknown'
    del movie['title']
    del movie['genre_names']

# Convert the data to a pandas DataFrame
data = pd.DataFrame(movies)

# Print out the first 10 lines to examine the data
print(data.head(1970))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['description'], data['genre'], test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Training the SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
