import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Load the preprocessed data
data = pd.read_pickle('preprocessed_data.pkl')

# Since data is already tokenized, join the tokens for TF-IDF vectorization
data['description'] = data['description'].apply(lambda x: ' '.join(x))

# Initialize the Vectorizer with current parameters
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 3))

# Split data into features and target
X = data['description']
y = data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71)

# Create and train the model
model = make_pipeline(vectorizer, MultinomialNB())
model.fit(X_train, y_train)

# Predict on the test set and evaluate
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1_scores = f1_score(y_test, y_pred, average=None)
recalls = recall_score(y_test, y_pred, average=None)
precisions = precision_score(y_test, y_pred, average=None)

# Print accuracy for the overall model
print("Overall Accuracy:", accuracy)

# Print metrics for each genre
print("\nMetrics for each genre:")
print(classification_report(y_test, y_pred, target_names=model.classes_))
