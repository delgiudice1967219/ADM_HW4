import math
import numpy as np
from scipy.sparse import csr_matrix

def char_vec(seen_movies, all_movies):
    '''
    Create a characteristic vector for a user's seen movies.
    Return the result as a sparse matrix for memory efficiency.
    '''
    all_movies = np.array(all_movies, dtype=int)

    # Create a binary vector (sparse format)
    vec = np.zeros(len(all_movies), dtype=int)

    for movie in seen_movies:
        # Locate the index of the movie in all_movies and set that position to 1
        movie_idx = np.where(all_movies == movie)[0]
        if movie_idx.size > 0:
            vec[movie_idx[0]] = 1
    
    # Return as a sparse matrix (compressed sparse row format)
    return csr_matrix(vec)


import random


def minhash(vec, num_hash_functions, coeff_a, coeff_b, prime):
    """
    Generate a MinHash signature for a given characteristic vector using NumPy.
    
    Parameters:
    - vec: A characteristic vector in sparse format (csr_matrix).
    - num_hash_functions: The number of hash functions to use.
    - coeff_a: Array of random coefficients 'a' for hash functions.
    - coeff_b: Array of random coefficients 'b' for hash functions.
    - prime: A prime number

    Returns:
    - sign: A NumPy array containing the MinHash signature, where each entry
            is the minimum hash value of indices corresponding to 1 in `vec`.
    """
    # Find indices where vec == 1
    nonzero_indices = vec.indices

    # If no indices have value 1, return infinity for all hash functions
    if len(nonzero_indices) == 0:
        return np.full(num_hash_functions, np.inf)
    
    # Compute hash values for all non-zero indices using vectorized operations
    hashes = (coeff_a[:, None] * nonzero_indices + coeff_b[:, None]) % prime    
    
    # Find the minimum hash value for each hash function
    sign = np.min(hashes, axis=1)
    
    return sign

def jaccard_sim(a , b):
    '''
    takes in imput two lists and computes the jaccard similarity between them
    '''
    set1 = set(a)
    set2 = set(b)
    intersection = len (set1.intersection(set2))
    union = len (set1.union(set2))

    return intersection/union


from collections import defaultdict

def lsh(signature_matrix, band_size):
    """
    Perform Locality-Sensitive Hashing on a signature matrix.
    
    Parameters:
        signature_matrix (numpy.ndarray): The MinHash signature matrix (rows=hashes, cols=items).
        band_size (int): The number of rows in each band.

    Returns:
        list of dict: Each element is a dictionary representing buckets for a band.
    """
    n_rows, n_cols = signature_matrix.shape
    assert n_rows % band_size == 0, "The number of rows must be divisible by the band size."
    
    num_bands = n_rows // band_size
    buckets = [defaultdict(list) for _ in range(num_bands)]  # One dictionary of buckets per band
    
    for band in range(num_bands):
        start_row = band * band_size
        end_row = start_row + band_size
        
        # Extract the band from the signature matrix
        band_data = signature_matrix[start_row:end_row, :]
        
        for col in range(n_cols):
            # Use tuple of band rows as the key
            band_signature = tuple(band_data[:, col])
            buckets[band][band_signature].append(col)  # Add column index to the bucket
            
    return buckets


def top2_query(user, signature_matrix, band_size, buckets, threshold):
    
    user_column = user-1
    similarity_count = defaultdict(int)

    for i, band in enumerate(buckets):
        band_signature = tuple(signature_matrix[i * band_size: (i + 1) * band_size, user_column])  # Get the user's signature for this band
        # Check for users in the same bucket
        for signature, items in band.items():
            if band_signature == signature:  # Compare the user's band signature to others in the bucket
                for item in items:
                    if item != user_column:  # Exclude the user themselves
                        similarity_count[item] += 1
    
    filtered_users = {user_id: count for user_id, count in similarity_count.items() if count > threshold}

    sorted_users = sorted(filtered_users.items(), key=lambda x: x[1], reverse=True)

    if not sorted_users:
        return f"No similar users found for user {user}. Adjust k, band_size"
    
    # Return the top 2 most similar users (convert to 1-based index)
    top_2_users = [user_id + 1 for user_id, _ in sorted_users[:2]]
    return top_2_users

    
    
def recommend_movie(user_seen_movies, result, df):
    """
    Recommend a movie based on the logic:
    - If both similar users have rated a movie, recommend this movie based on the average rating.
    - If there are no commonly rated movies, recommend the top-rated movies of the most similar user.
    
    Parameters:
        user_seen_movies (pd.DataFrame): DataFrame with a 'seen_movies' column for each user.
        result (list): Top 2 most similar users returned by the top2_query function.
        df (pd.DataFrame): DataFrame with movie ratings containing 'movieId', 'userId', and 'rating' columns.
        
    Returns:
        str: Recommended movie ID or a message if no movies can be recommended.
    """
    # Get seen movies for both similar users
    seen1 = set(user_seen_movies.iloc[result[0] - 1]['seen_movies'])
    seen2 = set(user_seen_movies.iloc[result[1] - 1]['seen_movies'])
    
    # Find common movies
    common_movies = seen1.intersection(seen2)
    
    if common_movies:
        # Compute average rating for each common movie
        common_movies = list(common_movies)  # Convert to list for filtering
        common_movie_ratings = (
            df[df['movieId'].isin(common_movies)]
            .groupby('movieId')['rating']
            .mean()
        )
        # Recommend the movie with the highest average rating
        recommended_movie = common_movie_ratings.idxmax()
        title = df.loc[df['movieId'] == recommended_movie, 'title'].values[0]
        return f"Recommended movie based on common ratings: Movie ID {recommended_movie}: '{title}' "
    
    else:
        # No common movies, recommend top-rated movies seen by the most similar user
        seen_by_top_user = list(seen1)  # Use the first similar user as the reference
        top_user_movies = (
            df[df['movieId'].isin(seen_by_top_user)]
            .groupby('movieId')['rating']
            .mean()
        )
        if top_user_movies.empty:
            return "No movies to recommend. Adjust parameters or input data."
        recommended_movie = top_user_movies.idxmax()
        title = df.loc[df['movieId'] == recommended_movie, 'title'].values[0]

        return f"Recommended movie based on top-rated movies of user {result[0]}: Movie ID {recommended_movie}: '{title}'"
