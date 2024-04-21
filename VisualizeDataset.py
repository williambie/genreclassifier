import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to present the dataset in a barplot

# Load dataset
data = pd.read_json('movies.json')

# Counting the genres
genre_counts = data['genre'].value_counts()

# Plotting the genre counts
plt.figure(figsize=(8, 4))
ax = sns.barplot(x=genre_counts.index,
                 y=genre_counts.values, palette='coolwarm')
plt.title('Number of movies in each genres')
plt.xlabel('Genre')
plt.ylabel('Movie count')
plt.xticks(rotation=45)

# Adding text labels on the bars
for i, value in enumerate(genre_counts.values):
    ax.text(i, value + 1, str(value), ha='center', va='bottom')

plt.show()
