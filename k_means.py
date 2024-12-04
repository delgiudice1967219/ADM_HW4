import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(point1, point2):
    '''
    Computes the Euclidean distance between two points

    Inputs: 
        point1, point2: coordinates about two points
    Outputs: 
        Euclidean distance between p1 and p2
    '''
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def closest_centroid(point, centroids):
    '''
    Find the nearest centroids to a given point

    Inputs:
        point: coordinates about a point
        centroids: list of coordinates one element for each centroid
    Outputs:
        the closest centroid for the given point
    '''
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances)


def kmeans(data, k, max_iterations=100, tolerance=1e-5):
    """
    Implementation of K-Means algorithm using MapReduce.

    Inputs:
        data: RDD object of points (each point is a list of coordinates).
        k: Number of cluster.
        max_iterations: Max number of iterations.
        tolerance: Convergence threshold for centroids.

    Outputs:
        centroids: List of final centroids.
        assignments: RDD object containing the belonging cluster for each point.
    """
    # Initialize the centroids
    # choose k random centroids from the data
    centroids = data.takeSample(False, k)

    for _ in range(max_iterations):
        # Assign each point to the closest centroid (Map Phase)
        clusters = data.map(lambda point: (closest_centroid(point, centroids), (point, 1)))

        # Compute the new centroids (Reduce Phase)
        # It return a list of tuples where each tuple represent: the first element is the index of the cluster
        # the second element are the coordinates of the new centroid assigned to that cluster
        new_centroids = (
            clusters
            .reduceByKey(lambda x, y: ([a + b for a, b in zip(x[0], y[0])], x[1] + y[1]))
            .mapValues(lambda x: [a / x[1] for a in x[0]])
            .collect()
        )
        
        # Extract only the centroids coordinates
        new_centroids = [centroid[1] for centroid in new_centroids]

        # If the centroids have been updated more than the tolerance value then update it
        # otherwise stop the algorithm, it reaches the convergence
        if all(euclidean_distance(c1, c2) < tolerance for c1, c2 in zip(centroids, new_centroids)):
            break

        # Update the centroids
        centroids = new_centroids

    # Return the final centroids and the assignation of each point
    assignments = data.map(lambda point: (closest_centroid(point, centroids), point))
    return centroids, assignments


def kmeans_plus_plus(data, k, max_iterations=100, tolerance=1e-5):
    """
    Implement K-Means++ using PySpark's RDD.

    Inputs:
        data: RDD of points (each point is a list of coordinates).
        k: Number of clusters.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance for centroid change.

    Outputs:
        centroids: Final list of centroids.
        assignments: RDD containing the cluster assignments of each point.
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

    # Proceed with standard K-means algorithm (assignment, update centroids)
    for _ in range(max_iterations):
        # Assign points to the closest centroid
        clusters = data.map(lambda point: (closest_centroid(point, centroids), (point, 1)))

        # Recompute the centroids
        new_centroids = (
            clusters
            .reduceByKey(lambda x, y: ([a + b for a, b in zip(x[0], y[0])], x[1] + y[1]))
            .mapValues(lambda x: [a / x[1] for a in x[0]])
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

    # Return the final centroids and the assignation of each point
    assignments = data.map(lambda point: (closest_centroid(point, centroids), point))
    return centroids, assignments


def plot_clustering(centroids, assignments):
    '''
    Create a plot to visualize clustering result, each cluster and the respective centroid

    Inputs:
        centroids: List of the final centroids.
        assignments: an RDD object containing each point with its respective cluster.
    Outputs:
        plot: the plot of the clustering result.
    '''
    # Define colors for each cluster
    cluster_colors = [
        (0, 128/255, 0),
        (54/255, 162/255, 235/255),
        (255/255, 159/255, 64/255),
        (128/255, 0, 128/255) 
    ]
    
    # Labels for clusters
    cluster_labels = [f"Cluster {i+1}" for i in range(len(cluster_colors))]

    # Extract coordinates of each point and their assegnation to the cluster
    points = np.array(assignments.map(lambda x: x[1]).collect())
    labels = assignments.map(lambda x: x[0]).collect()

    plt.figure(figsize=(8, 6))

    # Create a scatter for each cluster with a different color
    for i, color in enumerate(cluster_colors):
        # Get the points for the current cluster
        cluster_points = points[np.array(labels) == i]
        
        # Scatter the points for this cluster with the respective color
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=cluster_labels[i], s=50, alpha=0.8)

    # Add centroids
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

    plt.legend()
    plt.title('Clustering Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
