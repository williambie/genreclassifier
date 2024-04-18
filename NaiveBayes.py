import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Load the preprocessed data
data = pd.read_pickle('preprocessed_data.pkl')

# Since data is already tokenized, join the tokens for TF-IDF vectorization
data['description'] = data['description'].apply(lambda x: ' '.join(x))

# Best result yet: {"Accuracy": 0.40492957746478875, "Max Features": 4500, "Ngram Range": 1, "Random State": 71}
# Set your parameter lists
max_feature_list = [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 30000, 32500, 35000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 125000, 150000, 175000, 200000]
ngram_list = [1, 2, 3]
random_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
# Open a file to write the results to
with open('all_results.json', 'w') as f:
    for max_feats in max_feature_list:
        for ngram in ngram_list:
            for rand_state in random_list:
                # Initialize the Vectorizer with current parameters
                vectorizer = TfidfVectorizer(stop_words='english', max_features=max_feats, ngram_range=(1, ngram))

                # Split data into features and target
                X = data['description']
                y = data['genre']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

                # Create and train the model
                model = make_pipeline(vectorizer, MultinomialNB())
                model.fit(X_train, y_train)

                # Predict on the test set and evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Prepare the results dictionary
                results = {
                    'Accuracy': accuracy,
                    'Max Features': max_feats,
                    'Ngram Range': ngram,
                    'Random State': rand_state
                }

                # Write results to the file as a JSON string followed by a newline
                f.write(json.dumps(results) + '\n')

                # Optionally, print out the results to the console as well
                print(f"Results written for Max Features: {max_feats}, Ngram Range: {ngram}, Random State: {rand_state}")

