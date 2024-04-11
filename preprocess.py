import pandas as pd
import json
import re

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

# Convert the data to a pandas DataFrame
data = pd.DataFrame(movies)

# Print out the first 10 lines to examine the data
print(data.head(10))