import math
import numpy as np

def char_vec(seen_movies : set, all_movies_broadcast ):
    '''
    Create a characteristic vector for a user's seen movies.

    Parameters:
    - seen_movies (set): A set of movie IDs that the user has seen.
    - all_movies (set): A set of all possible movie IDs..

    Returns:
    - numpy.ndarray: A binary vector of length `len(all_movies)`, 
      where 1 indicates that the user has seen the movie and 0 indicates they have not.
    '''
    all_movies = all_movies_broadcast.value
    vector = [1 if movie in seen_movies else 0 for movie in all_movies]
    return vector




def create_signature(vec, hash_functions):
    '''
    Generate a MinHash signature for a given characteristic vector.

    Parameters:
    - vec: A characteristic vector (binary list) representing a set.
    - hash_functions: A list of hash functions to apply.

    Returns:
    - sign: A list containing the MinHash signature, where each entry
            is the minimum hash value of indices corresponding to 1 in vec.
    '''
    k = len(hash_functions)
    sign = [math.inf]*k #initialize to inf
    for i in range(k):
        for indx, value in enumerate(vec):
            if value == 1:  # Update if value is 1
                if hash_functions[i](indx) < sign[i]:  # Apply the i-th hash function
                        sign[i] = hash_functions[i](indx)
    return sign


