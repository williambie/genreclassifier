import pandas as pd
import json
import re
import nltk
from nltk.tokenize import word_tokenize

# Load the data from the json file
with open('movies.json', 'r') as file:
    movies = json.load(file)

# Preprocess the movie data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

# Preprocess the movie plot and genre names and remove the title
for movie in movies:
    movie['description'] = preprocess_text(movie['description'])

# Convert the data to a pandas DataFrame
data = pd.DataFrame(movies)

data.to_pickle('preproccesed_data.pkl')

# Print out the first 10 lines to examine the data
print(data.head(10))