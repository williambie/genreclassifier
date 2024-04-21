import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Define the endpoint and API key for The Movie Database API
API_KEY = os.getenv("API_KEY")
URL_BASE = "https://api.themoviedb.org/3/"
MOVIE_LIST_ENDPOINT = f"{URL_BASE}discover/movie?api_key={API_KEY}"

# Fetch the total number of pages available
def get_total_pages():
    response = requests.get(MOVIE_LIST_ENDPOINT)
    if response.status_code == 200:
        return response.json().get('total_pages')
    return 0

# Get all movies from a specific page
def get_movies_from_page(page_number):
    response = requests.get(f"{MOVIE_LIST_ENDPOINT}&page={page_number}")
    if response.status_code == 200:
        movies_data = response.json().get('results', [])
        filtered_movies = [
            {
                'description': movie.get('overview'),
                'genres': [movie.get('genre_ids')[0]] if movie.get('genre_ids') else [],
            }
            for movie in movies_data
            if movie.get('overview') and movie.get('genre_ids')
        ]
        return filtered_movies
    return []

# Saving the data to a JSON file
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Main function to fetch and save the data
def main():
    total_pages = get_total_pages()

    # Limit 20 000 pages
    max_pages = min(total_pages, 20000)

    all_movies = []

    for page in range(1, max_pages + 1):
        print(f"Fetching page {page} of {max_pages}...")
        all_movies.extend(get_movies_from_page(page))

    save_to_json(all_movies, 'test.json')
    print("Saved to test.json")

main()

# Mapping genre IDs to genre names
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

# Function for replacing genre IDs with genre names
def replace_genre_ids_with_names(movie_data):
    for movie in movie_data:
        if 'genres' in movie and movie['genres']:
            try:
                movie['genre'] = genre_id_to_name.get(movie['genres'][0])
            except KeyError as e:
                print(f"Warning: Genre ID {e} not found in mapping. Skipping.")
            del movie['genres']

# Tries openeing the JSON file and loads the data into a variable called movies_data 
try:
    file_path = 'test.json'
    with open(file_path, 'r') as file:
        movies_data = json.load(file)
except FileNotFoundError:
    print("Error: The specified JSON file was not found.")
    exit(1)

replace_genre_ids_with_names(movies_data)

# Tries writing the updated JSON data to the file
try:
    output_file_path = 'test.json'
    with open(output_file_path, 'w') as file:
        json.dump(movies_data, file, indent=2)
except IOError as e:
    print(f"Error writing to output file: {e}")
    exit(1)

print("The JSON data has been updated with genre names.")