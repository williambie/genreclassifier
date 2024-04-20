import json

# Load JSON data
with open('movies.json', 'r') as file:
    movies = json.load(file)

# Use a set to remove duplicates and check for empty fields
seen_descriptions = set()
unique_movies = []
for movie in movies:
    if movie['description'] and movie['genre'] and movie['description'] not in seen_descriptions:
        unique_movies.append(movie)
        seen_descriptions.add(movie['description'])

# Save the cleaned data back to JSON
with open('movies.json', 'w') as file:
    json.dump(unique_movies, file, indent=4)

print(f"Removed duplicates and movies with empty fields. Total unique valid movies: {len(unique_movies)}")
