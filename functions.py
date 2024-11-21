import math

def char_vec(seen_movies, all_movies):
    '''
    Create a characteristic vector for a user's seen movies.
    For each movie ID in all_movies, add 1 to the vector if the movie
    is in the user's seen_movies list, otherwise add 0.
    '''
    v = []
    for i in all_movies:
        if i in seen_movies:
            v.append(1)
        else:
            v.append(0)
    return v

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