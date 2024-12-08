import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    """
    Create a linkage matrix from a hierarchical clustering model and plot the dendrogram.

    Parameters:
        model: Fitted AgglomerativeClustering model
        **kwargs: Additional keyword arguments to customize the dendrogram (e.g., `color_threshold`,
                `leaf_rotation`, etc...).

    Outputs:
        dendrogram: A dendrogram plot based on the just built linkage matrix
    """
    # Initialize an array to store the number of samples (data points) under each node in the hierarchy
    counts = np.zeros(model.children_.shape[0])
    # Total number of samples in the dataset
    n_samples = len(model.labels_)

    # Loop over each pair of merged clusters in the hierarchical model
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                # If the index is less than the number of samples, it corresponds to a leaf node (a single sample)
                current_count += 1
            else:
                # If the index is greater than or equal to n_samples, it corresponds to a merged node
                # Add the count of samples under the child node
                current_count += counts[child_idx - n_samples]
        # Store the total count of samples under this merged node
        counts[i] = current_count

    # Create the linkage matrix indicating the
    # indices of the merged clusters at each step
    # distance between the clusters being merged
    # number of samples under each cluster after merging
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the dendrogram using the linkage matrix
    return dendrogram(linkage_matrix, **kwargs)


def euclidean_distance(point1, point2):
    """
    Computes the Euclidean distance between two points

    Inputs:
        point1, point2: coordinates about two points
    Outputs:
        Euclidean distance between p1 and p2
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def closest_centroid(point, centroids):
    """
    Find the nearest centroids to a given point

    Inputs:
        point: coordinates about a point
        centroids: list of coordinates one element for each centroid
    Outputs:
        Returns the closest centroid for the given point
    """
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances)


def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    """
    Implementation of K-Means algorithm using MapReduce.

    Inputs:
        data: RDD of points (each point is a list of coordinates).
        k: Number of cluster.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance for centroids update.

    Outputs:
        centroids_history: List of centroids at each iteration.
        assignments_history: List of assignments for each iteration.
    """
    # Initialize the centroids
    # select k random centroids from the data points
    # False, stay because we want it without replacement
    centroids = data.takeSample(False, k)

    # Lists to store history of centroids and assignments
    # Store the first random centroids assignment
    centroids_history = [centroids]
    assignments_history = []

    # Assign the first set of points to the nearest centroid
    initial_assignments = data.map(lambda point: (closest_centroid(point, centroids), point))

    # Store the assignments for all the points for the first iteration
    assignments_history.append(initial_assignments.collect())

    # Loop over each iteration up to max_iterations
    for _ in range(max_iterations):
        # Assign each point to the closest centroid (Map Phase)
        # Computes the index of the closest centroid for each data point
        # It return an RDD of tuples, where in each tuple the first element is the index of the closest centroid 
        # The second element is a tuple containing the point's coordinates
        clusters = data.map(lambda point: (closest_centroid(point, centroids), point))

        # Compute the new centroids (Reduce Phase)
        # It return a list of tuples where in each tuple the first element is the index of the cluster
        # the second element are the coordinates of the new centroid assigned to that cluster
        new_centroids = (
            clusters
            .mapValues(lambda x: (x, 1))  #Add a counter 1 to each point, transforming each point into a pair ([coordinates], 1)
            .reduceByKey(lambda a, b: ([a[0][i] + b[0][i] for i in range(len(a[0]))], a[1] + b[1]))  # Aggregate coordinates and counts
            .mapValues(lambda x: [val / x[1] for val in x[0]])  # Compute average (new centroid)
            .collect() # Collect results as a list
        )
        
        # Extract only the centroids coordinates from the list of tuples.
        # Each tuple contains the centroid index and its new coordinates
        new_centroids = [centroid[1] for centroid in new_centroids]

        # If the centroids have been updated more than the tolerance value then update it
        # otherwise stop the algorithm, it reaches the convergence
        if all(euclidean_distance(c1, c2) < tolerance for c1, c2 in zip(centroids, new_centroids)):
            break

        # Update the centroids
        centroids = new_centroids

        # Store assignments and centroids for the current iteration
        assignments_history.append(clusters.collect())  # Collect current assignments
        centroids_history.append(centroids)  # Store new centroids

    return centroids_history, assignments_history


def kmeans_plus_plus(data, k, max_iterations=100, tolerance=1e-4):
    """
    Implementation of K-Means++ algorithm using MapReduce.

    Inputs:
        data: RDD of points (each point is a list of coordinates).
        k: Number of clusters.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance for centroid change.

    Outputs:
        centroids_history: List of centroids at each iteration.
        assignments_history: List of assignments for each iteration.
    """

    # Randomly pick the first centroid
    # Using [0] to take only the first and only element from the list of sampled data
    centroids = [data.takeSample(False, 1)[0]]

    # Select the remaining k-1 centroids using a distance-based probability distribution
    for _ in range(k - 1):
        distances = data.map(lambda point: min([euclidean_distance(point, c) for c in centroids]))
        total_distance = distances.reduce(lambda a, b: a + b)
        probabilities = distances.map(lambda d: d ** 2 / total_distance)

        # Choose the next centroid based on these probabilities
        r = np.random.random()
        cumulative_prob = 0.0
        for point, prob in zip(data.collect(), probabilities.collect()):
            cumulative_prob += prob
            if cumulative_prob >= r:
                centroids.append(point)
                break

    centroids_history = [centroids]
    assignments_history = []

    # Assign the first set of points to the nearest centroid
    initial_assignments = data.map(lambda point: (closest_centroid(point, centroids), point))

    # Store the assignments for all the points for the first iteration
    assignments_history.append(initial_assignments.collect())

    # Proceed with standard K-means algorithm (assignment, update centroids)
    for _ in range(max_iterations):
        # Assign points to the closest centroid
        clusters = data.map(lambda point: (closest_centroid(point, centroids), point))

        # Compute the new centroids
        new_centroids = (
            clusters
            .mapValues(lambda x: (x, 1))
            .reduceByKey(lambda a, b: ([a[0][i] + b[0][i] for i in range(len(a[0]))], a[1] + b[1]))
            .mapValues(lambda x: [val / x[1] for val in x[0]])
            .collect()
        )

        # Extract only the centroids coordinates
        new_centroids = [centroid[1] for centroid in new_centroids]

        # If the centroids have been updated more than the tolerance value then update it
        # otherwise stop the algorithm, it reaches the convergence
        if all(euclidean_distance(c1, c2) < tolerance for c1, c2 in zip(centroids, new_centroids)):
            break
        
        # Update centroids
        centroids = new_centroids

        # Store the assignments and centroids for this iteration
        assignments_history.append(clusters.collect())  # Store assignments for the current iteration
        centroids_history.append(centroids)  # Store new centroids for the current iteration

    return centroids_history, assignments_history


def plot_clustering(centroids, assignments):
    """
    Create a plot to visualize clustering result, each cluster and the respective centroid

    Inputs:
        centroids: List of the final centroids.
        assignments: an RDD object containing each point with its respective cluster.

    Outputs:
        plot: the plot of the clustering result.
    """
    # Define colors for each cluster
    cluster_colors = [
        (0, 128/255, 0),
        (54/255, 162/255, 235/255),
        (255/255, 159/255, 64/255),
        (128/255, 0, 128/255) 
    ]
    
    # Labels for clusters
    cluster_labels = [f"Cluster {i+1}" for i in range(len(cluster_colors))]

    # Extract coordinates of each point and their assignation to the cluster
    labels = np.array([assignment[0] for assignment in assignments])  # Extract cluster labels
    points = np.array([assignment[1] for assignment in assignments])  # Extract point coordinates

    plt.figure(figsize=(8, 6))

    # Create a scatter plot for each cluster with a different color
    for i, color in enumerate(cluster_colors):
        # Get the points for the current cluster
        cluster_points = points[labels == i]

        # Plot the points for this cluster with the respective color
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=color,
            label=cluster_labels[i],
            s=50,
            alpha=0.8
        )

    # Add centroids
    centroids = np.array(centroids)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='red',
        marker='x',
        s=200,
        label='Centroids'
    )
    plt.legend()
    plt.title('Clustering Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()