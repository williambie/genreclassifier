import numpy as np
import pandas as pd
import nltk
import json
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


# Load the data from the json file
with open('movies.json', 'r') as file:
  data = json.load(file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

for movie in data:
    movie['description'] = preprocess_text(movie['description'])
    movie['genre_names'] = [genre.lower() for genre in movie['genre_names']]
    del movie['title']
    
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

print(df.head())
