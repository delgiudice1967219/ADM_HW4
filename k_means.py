import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(point1, point2):
    '''
    Computes the Euclidean distance between two points
    Inputs: point1, point2 coordinates about two points
    Outputs: Euclidean distance between p1 and p2
    '''
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def closest_centroid(point, centroids):
    '''
    Find the nearest centroids to a given point
    Inputs:
        point: coordinates about a point
        centroids: list of coordinates of centroids
    Outputs: the closest centroid for the given point
    '''
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances)


def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    """
    Implementation of K-Means algorithm using MapReduce.

    Inputs:
        data: RDD object of points (each point is a tuple of coordinates).
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

    for i in range(max_iterations):
        # Assign each point to the nearest centroid (Map Phase)
        clusters = data.map(lambda point: (closest_centroid(point, centroids), (point, 1)))

        # Compute the new centroids (Reduce Phase)
        new_centroids = (
            clusters
            .reduceByKey(lambda x, y: ([a + b for a, b in zip(x[0], y[0])], x[1] + y[1]))
            .mapValues(lambda x: [a / x[1] for a in x[0]])
            .collect()
        )

        # Converte i nuovi centroidi in una lista ordinata
        new_centroids = sorted(new_centroids, key=lambda x: x[0])
        new_centroids = [centroid[1] for centroid in new_centroids]

        # If the centroids have been updated more than the tolerance value then update it
        # otherwise stop the algorithm
        if all(euclidean_distance(c1, c2) < tolerance for c1, c2 in zip(centroids, new_centroids)):
            break

        centroids = new_centroids

    # Return the final centroids and the assignation of each point
    assignments = data.map(lambda point: (closest_centroid(point, centroids), point))
    return centroids, assignments


def kmeans_plus_plus(data, k, max_iterations=100, tolerance=1e-3):
    """
    Implement K-Means++ using PySpark's RDD.

    Args:
        data: RDD of points (each point is a list or tuple of coordinates).
        k: Number of clusters.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance for centroid change.

    Returns:
        centroids: Final list of centroids.
        assignments: RDD containing the cluster assignments of each point.
    """

    # Step 1: Randomly pick the first centroid
    centroids = [data.takeSample(False, 1)[0]]

    # Step 2: Select the remaining k-1 centroids using a distance-based probability distribution
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

    # Step 5: Proceed with standard K-means algorithm (assignment, update centroids)
    for i in range(max_iterations):
        # Step 6: Assign points to the nearest centroid
        clusters = data.map(lambda point: (closest_centroid(point, centroids), (point, 1)))

        # Step 7: Recompute the centroids
        new_centroids = (
            clusters
            .reduceByKey(lambda x, y: ([a + b for a, b in zip(x[0], y[0])], x[1] + y[1]))
            .mapValues(lambda x: [a / x[1] for a in x[0]])
            .collect()
        )

        # Convert the new centroids into a list sorted by the cluster index
        new_centroids = sorted(new_centroids, key=lambda x: x[0])
        new_centroids = [centroid[1] for centroid in new_centroids]

        # Step 8: Check for convergence (if centroids don't change significantly)
        if all(euclidean_distance(c1, c2) < tolerance for c1, c2 in zip(centroids, new_centroids)):
            break

        centroids = new_centroids  # Update centroids

    # Step 9: Return final centroids and the assignments
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
    # Estrai i punti e le assegnazioni dal driver
    points = np.array(assignments.map(lambda x: x[1]).collect())  # Raccogli i punti nel driver
    labels = assignments.map(lambda x: x[0]).collect()  # Raccogli le etichette nel driver

    # Converti i punti in un array NumPy per facilitarne la manipolazione
    #points = np.array(points)

    # Creazione dello scatterplot
    plt.figure(figsize=(8, 6))

    # Add each point with a different colour based on the belonging cluster
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', s=50)

    # Add centroids to the plot
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

    plt.title('Clustering Result')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.colorbar(scatter)
    plt.show()