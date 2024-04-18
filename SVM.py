import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
from sklearn.exceptions import UndefinedMetricWarning

nltk.download('wordnet')
nltk.download('stopwords')

# Suppress warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

# Load and preprocess data
data = pd.read_pickle('preprocessed_data.pkl')
data['description'] = data['description'].apply(lambda x: ' '.join(x))
data['description'] = data['description'].apply(lemmatize_text)

X = data['description']
y = data['genre']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline setup
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('svc', SVC())
])

# {"Best Parameters": {"svc__C": 1, "svc__kernel": "linear", "tfidf__max_features": 10000, "tfidf__ngram_range": [1, 1]}, "Accuracy": 0.3933601609657948}
# Parameter grid
param_grid = {
    'tfidf__max_features': [10000, 20000],
    'tfidf__ngram_range': [(1, 3)],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save results
results = {
    'Best Parameters': grid_search.best_params_,
    'Classification Report': classification_report(y_test, y_pred, output_dict=True)
}

with open('svm_optimal_results.json', 'w') as f:
    json.dump(results, f)

print("Optimized results and parameters saved.")
