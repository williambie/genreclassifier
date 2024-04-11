import json

# Mapping from genre IDs to genre names
genre_id_to_name = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

def replace_genre_ids_with_names(movie_data):
    for movie in movie_data:
        if 'genres' in movie:  # Adjusted to match your JSON key
            # Handle missing genre IDs with a try-except block
            try:
                movie['genre_names'] = [genre_id_to_name[genre_id] for genre_id in movie['genres']]  # Adjusted to match your JSON key
            except KeyError as e:
                print(f"Warning: Genre ID {e} not found in mapping. Skipping.")
            del movie['genres']  # Adjusted to remove the original genres field

try:
    file_path = 'movies.json'
    with open(file_path, 'r') as file:
        movies_data = json.load(file)
except FileNotFoundError:
    print("Error: The specified JSON file was not found.")
    exit(1)

replace_genre_ids_with_names(movies_data)

try:
    output_file_path = 'movies.json'
    with open(output_file_path, 'w') as file:
        json.dump(movies_data, file, indent=2)
except IOError as e:
    print(f"Error writing to output file: {e}")
    exit(1)

print("The JSON data has been updated with genre names.")
