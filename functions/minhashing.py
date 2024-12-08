import numpy as np
from scipy.sparse import csr_matrix


def char_vec(seen_movies, all_movies):
    """
    Create a characteristic vector for a user's seen movies.
    Return the result as a sparse matrix to save memory for large datasets.

    Parameters:
    - seen_movies (list or array-like): A list of movie IDs the user has watched.
    - all_movies (list or array-like): A list of all unique movie IDs in the dataset.

    Returns:
    -scipy.sparse.csr_matrix: A sparse binary vector of size len(all_movies), where:
        - Each position corresponds to a movie in all_movies.
        - The value is 1 if the movie is in seen_movies, otherwise 0.
    """
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
