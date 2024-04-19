import json
import requests

# Constants
API_KEY = '018ff1d3840698b0ab96703e3d1ee53d' # Dont mind this key, it's a dummy key :)
URL_BASE = "https://api.themoviedb.org/3/"
MOVIE_LIST_ENDPOINT = f"{URL_BASE}discover/movie?api_key={API_KEY}"

def get_total_pages():
    response = requests.get(MOVIE_LIST_ENDPOINT)
    if response.status_code == 200:
        return response.json().get('total_pages')
    return 0


def get_movies_from_page(page_number):
    response = requests.get(f"{MOVIE_LIST_ENDPOINT}&page={page_number}")
    if response.status_code == 200:
        movies_data = response.json().get('results', [])
        # Filter movies and include only the first genre_id if available
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

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    total_pages = get_total_pages()

    # Limit 20 000 pages
    max_pages = min(total_pages, 20000)

    all_movies = []

    for page in range(1, max_pages + 1):
        print(f"Fetching page {page} of {max_pages}...")
        all_movies.extend(get_movies_from_page(page))

    save_to_json(all_movies, 'movies.json')
    print("Saved to movies.json")

main()

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
        # Check if 'genres' exists and is not empty
        if 'genres' in movie and movie['genres']:
            try:
                # Directly assign the genre name string instead of a list
                movie['genre'] = genre_id_to_name.get(movie['genres'][0])
            except KeyError as e:
                print(f"Warning: Genre ID {e} not found in mapping. Skipping.")
            del movie['genres']  # Remove the original 'genres' field

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