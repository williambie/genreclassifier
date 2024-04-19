import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_json('movies.json')

# Counting the genres
genre_counts = data['genre'].value_counts()

# Plotting the genre counts
plt.figure(figsize=(8, 4))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='coolwarm')
plt.title('Count of Selected Genres in Dataset')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()