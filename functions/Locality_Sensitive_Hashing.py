from collections import defaultdict


def lsh(signature_matrix, band_size):
    """
    Perform Locality-Sensitive Hashing on a signature matrix.
    
    Parameters:
        signature_matrix (numpy.ndarray): The MinHash signature matrix (rows=hashes, cols=items).
        band_size (int): The number of rows in each band.

    Returns:
        list of dict: Each element is a dictionary representing buckets for a band.
    
    Function Description: - The function performs Locality-Sensitive Hashing by dividing the signature matrix into
    bands of size `band_size`.
    - For each band, the rows are grouped together to form a signature, which is used as
    the key in the corresponding bucket.
    - Items (columns in the signature matrix) that share the same signature are
    added to the same bucket.
    - The resulting `buckets` list contains one dictionary per band, where each dictionary
    maps band signatures to a list of item indices with that signature.
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
    """
    Identify the two most similar users to the given user based on LSH buckets and MinHash signatures.

    Parameters:
    -user (int): The ID of the user for whom similar users are to be identified.
    -signature_matrix (numpy.ndarray): The MinHash signature matrix with rows as hash functions and columns as users.
    -band_size (int): The number of rows in each band for LSH.
    -buckets (list of dict): The bucket structure created during LSH, mapping band signatures to user indices.
    -threshold (int): The minimum number of common buckets required to consider users as similar.

    Returns:
        list of int: The IDs of the top 2 most similar users, in 1-based index format.
        str: A message indicating no similar users were found if the result is empty.

    Function Description:
    - The function computes the "band signature" for the target user across all bands.
    - It then compares this signature with others in the same bucket to count the occurrences of similar users.
    - Only users with a similarity count exceeding the threshold are considered.
    - The users are sorted based on their similarity count, and the top two similar users are returned.
    - If no users meet the threshold, a message suggesting parameter adjustment is returned.
    """

    user_column = user - 1
    similarity_count = defaultdict(int)

    for i, band in enumerate(buckets):
        band_signature = tuple(
            signature_matrix[i * band_size: (i + 1) * band_size, user_column])  # Get the user's signature for this band
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


def recommend_movie(user_seen_movies, similar_users, df):
    """
    Recommend a movie based on the logic:
    - If both similar users have rated a movie, recommend this movie based on the average rating.
    - If there are no commonly rated movies, recommend the top-rated movies of the most similar user.
    
    Parameters:
        user_seen_movies (pd.DataFrame): DataFrame with a 'seen_movies' column for each user.
        similar_users (list): Top 2 most similar users returned by the top2_query function.
        df (pd.DataFrame): DataFrame with movie ratings containing 'movieId', 'userId', and 'rating' columns.
        
    Returns:
        str: Recommended movie ID or a message if no movies can be recommended.
    """

    seen1 = set(user_seen_movies.iloc[similar_users[0] - 1]['seen_movies'])
    seen2 = set(user_seen_movies.iloc[similar_users[1] - 1]['seen_movies'])
    common_movies = seen1.intersection(seen2)

    if common_movies:
        # Calculate the average rating only for the two similar users
        common_ratings = df[df['movieId'].isin(common_movies)]
        common_ratings = common_ratings[common_ratings['userId'].isin(similar_users)]
        avg_ratings = common_ratings.groupby('movieId')['rating'].mean().reset_index()
        avg_ratings = avg_ratings.sort_values(by='rating', ascending=False).head(5)

        # Fetch titles for the top movies
        top_movies = avg_ratings.merge(df[['movieId', 'title']].drop_duplicates(), on='movieId', how='left')
        top_movies = top_movies.rename(columns={'rating': 'avg_rating'})

        # Use tabulate to format the DataFrame
        out = "Top 5 recommended movies based on common ratings:"
        return out, top_movies

    else:
        # If no common movies, recommend the top movies of the first similar user
        first_user_movies = user_seen_movies.iloc[similar_users[0] - 1]['seen_movies']
        first_user_ratings = df[df['movieId'].isin(first_user_movies)]
        first_user_ratings = first_user_ratings[first_user_ratings['userId'] == similar_users[0]]
        top_movies = first_user_ratings.sort_values(by='rating', ascending=False).head(5)[
            ['movieId', 'rating', 'title']]
        top_movies = top_movies.rename(columns={'rating': 'avg_rating'})

        out = "No common movies found. Top 5 movies rated by user {similar_users[0]}"
        return out, top_movies
