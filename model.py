import numpy as np
import pandas as pd
import nltk
import json
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from collections import Counter

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
    del movie['title']

# Convert the data to a pandas DataFrame
data = pd.DataFrame(movies)

# Print out the first 10 lines to examine the data
print(data.head(10))

# Section for visualizing the most frequent genres
all_genres = [genre for sublist in data['genre_names'] for genre in sublist]
genre_counts = Counter(all_genres)
genres, counts = zip(
    *sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(10, 6))
plt.bar(genres, counts, color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.title('Frequency of Movie Genres')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()