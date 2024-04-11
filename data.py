import json
import requests

# Constants
API_KEY = '018ff1d3840698b0ab96703e3d1ee53d'
URL_BASE = "https://api.themoviedb.org/3/"
MOVIE_LIST_ENDPOINT = f"{URL_BASE}discover/movie?api_key={API_KEY}"  # Corrected the URL formation

def get_total_pages():
    response = requests.get(MOVIE_LIST_ENDPOINT)
    if response.status_code == 200:
        return response.json().get('total_pages')
    return 0

def get_movies_from_page(page_number):
    response = requests.get(f"{MOVIE_LIST_ENDPOINT}&page={page_number}")
    if response.status_code == 200:
        movies_data = response.json().get('results', [])
        # Filter only the required fields
        filtered_movies = [
            {
                'title': movie.get('title'),
                'description': movie.get('overview'),
                'genres': movie.get('genre_ids'),  # This will return genre IDs. Consider a lookup function for genre names if necessary
            }
            for movie in movies_data
        ]
        return filtered_movies
    return []

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    total_pages = get_total_pages()

    # Limit to 10 pages as the comment says 500, but the code does 10
    max_pages = min(total_pages, 1)

    all_movies = []

    for page in range(1, max_pages + 1):
        print(f"Fetching page {page} of {max_pages}...")
        all_movies.extend(get_movies_from_page(page))

    save_to_json(all_movies, 'movies.json')
    print("Saved to movies.json")

if __name__ == "__main__":
    main()
