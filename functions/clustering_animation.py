import plotly.graph_objects as go


def kmeans_animation(assignments_history, centroids_history, cluster_colors=None):
    """
    Visualize K-Means progression of the cluster assignments at each iteration.

    Inputs:
        assignments_history: List of lists containing cluster assignments at each iteration.
                            Each sublist contain tuples (cluster_index, coordinates).
        centroids_history: List of lists containing centroid coordinates at each iteration.
                            Each sublist contain tuples (x, y).
        cluster_colors: List of colors (in RGBA format) for the clusters. If None, default colors are used.

    Outputs:
        fig: A Plotly figure object with animation for clustering iterations.
    """
    # Default cluster colors if none provided
    # if more than 4 cluster are present, add other color
    if cluster_colors is None:
        cluster_colors = [
            'rgba(0, 128, 0, 0.8)',  # Green
            'rgba(54, 162, 235, 0.8)',  # Blue
            'rgba(255, 159, 64, 0.8)',  # Orange
            'rgba(128, 0, 128, 0.8)'  # Purple
        ]

    # Generate labels for each cluster (i.e., "Cluster 1", "Cluster 2", etc...)
    cluster_labels = [f"Cluster {i + 1}" for i in range(len(cluster_colors))]

    # Create traces for the legend (one for each cluster)
    legend_traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            # Legend entry name, number of the cluster (i.e. "Cluster 1")
            name=cluster_labels[i],
            # Marker size and color
            marker=dict(size=8, color=cluster_colors[i])
        )
        for i in range(len(cluster_colors))
    ]

    # One frame for each iteration of K-Means
    # Initialize the frames list where store iterations
    # Each iteration has a different plot that will be animated
    frames = []
    for i in range(len(centroids_history)):
        # Get the cluster colors and assignment for each point in this iteration
        colors = [cluster_colors[p[0]] for p in assignments_history[i]]
        labels = [cluster_labels[p[0]] for p in assignments_history[i]]
        # X coordinates of points
        points_x = [p[1][0] for p in assignments_history[i]]
        # Y coordinates of points
        points_y = [p[1][1] for p in assignments_history[i]]

        # Create a scatter plot for the points in this iteration
        frame_data = [
            go.Scatter(
                x=points_x,
                y=points_y,
                mode='markers',
                # Name of each point
                name="Movie",
                # Marker properties
                marker=dict(size=8, color=colors),
                # Hover template, information we can see passing on each point
                hovertemplate=(
                    # Cluster label
                    '%{customdata}<br>'
                    # X coordinate
                    'X: %{x}<br>'
                    # Y coordinate
                    'Y: %{y}'
                ),
                customdata=labels,
                # Don't display this trace in the legend, overlapping information
                showlegend=False
            )
        ]

        # Create a scatter plot for the centroids in this iteration
        # X coordinates of centroids
        centroids_x = [c[0] for c in centroids_history[i]]
        # Y coordinates of centroids
        centroids_y = [c[1] for c in centroids_history[i]]

        # Append the frame for centroids visualization
        frame_data.append(
            go.Scatter(
                x=centroids_x,
                y=centroids_y,
                mode='markers',
                name="Centroids",
                marker=dict(size=12, color='red', symbol='x'),
                hovertemplate=(
                    'Centroid<br>'
                    'X: %{x}<br>'
                    'Y: %{y}'
                ),
                # Show in the legend, centroids marker will always be the same
                showlegend=True
            )
        )

        # Append frame, storing also the iteration number
        frames.append(go.Frame(data=frame_data, name=f"Iteration {i + 1}"))

    # First assignment(initial data)
    # image displayed when the animation is stopped
    # repeat the process just for the first assignment
    initial_colors = [cluster_colors[p[0]] for p in assignments_history[0]]
    initial_labels = [cluster_labels[p[0]] for p in assignments_history[0]]
    initial_points_x = [p[1][0] for p in assignments_history[0]]
    initial_points_y = [p[1][1] for p in assignments_history[0]]

    initial_data = [
        go.Scatter(
            x=initial_points_x,
            y=initial_points_y,
            mode='markers',
            name="Movie",
            marker=dict(size=8, color=initial_colors),
            hovertemplate=(
                '%{customdata}<br>'
                'X: %{x}<br>'
                'Y: %{y}'
            ),
            customdata=initial_labels,
            showlegend=False
        ),
        go.Scatter(
            x=[c[0] for c in centroids_history[0]],
            y=[c[1] for c in centroids_history[0]],
            mode='markers',
            name="Centroids",
            marker=dict(size=12, color='red', symbol='x'),
            hovertemplate=(
                'Centroid<br>'
                'X: %{x}<br>'
                'Y: %{y}'
            )
        )
    ]

    # Add legend traces to initial data
    initial_data.extend(legend_traces)

    # Create figure with data, layout, and animation controls
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            # Title
            title="K-Means Clustering Over Iterations",
            # X label
            xaxis=dict(title="PC1", showgrid=True, autorange=True),
            # Y label
            yaxis=dict(title="PC2", showgrid=True, autorange=True),
            # Figure width
            width=1000,
            # Figure height
            height=800,
            # Play and Pause buttons
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        # Play button
                        dict(
                            label="Play",
                            method="animate",
                            # 1000ms duration (1 second)
                            # redraw and fromcurrent forces to change dynamically
                            # basing on the previous frame without starting from the first one
                            args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]
                        ),
                        # Pause button
                        dict(
                            label="Pause",
                            method="animate",
                            # immediate: the animation should stop immediately
                            # stop the progression at the next frame, 0 duration
                            # do not redraw, no frame changes during pause
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                        )
                    ]
                )
            ],
            annotations=[
                dict(
                    text="Iteration: 1",
                    x=0.5,
                    y=1.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        ),
        # Display each iteration frame
        frames=[
            go.Frame(
                data=frame.data,
                # Frame name (iteration number)
                name=f"Iteration {i + 1}",
                # Layout for the frame (iteration number annotation)
                layout=go.Layout(
                    annotations=[
                        dict(
                            # Annotation text for each frame
                            text=f"Iteration: {i + 1}",
                            # Position
                            x=0.5,
                            y=1.15,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16)
                        )
                    ]
                )
            )
            for i, frame in enumerate(frames)   # Loop through all frames
        ]
    )

    # Display the figure
    fig.show()
