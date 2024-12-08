# ADM - Homework 4: Movie Recommendation System, Group #10

This GitHub repository contains the implementation of the fourth homework of the **Algorithmic Methods of Data Mining** course for the master's degree in Data Science at Sapienza (2024-2025). This homework was completed by Group #10. The details of the assignement are specified here:  https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_4

**Team Members:**
* Xavier Del Giudice, 1967219, delgiudice.1967219@studenti.uniroma1.it
* Valeria Avino, 1905974, avino.1905974@studenti.uniroma1.it
* Mattia Mungo, 1883175, mungo.1883175@studenti.uniroma1.it
* Davide Vitale, 1794386, vitale.1794386@studenti.uniroma1.it

The ```main.ipynb``` with the main script can be visualized in the jupyter notebook viewer: [nbviewer](https://nbviewer.org/github/delgiudice1967219/ADM_HW4/blob/main/main.ipynb)

## Repository Structure

```
├── functions/                                      # Directory containing core project modules
│   ├── Locality_Sensitive_Hashing.py               # Implementation of LSH for recommendation
│   ├── clustering.py                               # Custom clustering methods
│   ├── clustering_animation.py                     # Code for visualizing clustering progress
│   ├── preprocessing.py                            # Helper utilities for preprocessing
│   ├── minhashing.py                               # MinHash implementation for user similarities
│   └── tools.py                                    # Helper functions for find similarity
├── main.ipynb                                      # Main notebook containg all the runned cells
├── .gitignore                                      # File to specify files and directories ignored by Git
├── README.md                                       # Project documentation
└── LICENSE                                         # License file for the project
```

Here are links to all the files:
- [main.ipynb](main.ipynb): The main notebook containing implementation and results
- [functions](functions): Contains modularized scripts for the homework
  * [Locality_Sensitive_Hashing.py](functions/Locality_Sensitive_Hashing.py): Implements LSH to find user similarities
  * [clustering.py](functions/clustering.py): Contains clustering logic, including K-means and K-means++
  * [clustering_animation.py](functions/clustering_animation.py): Scripts for generating iterative clustering visualizations
  * [preprocessing.py](functions/preprocessing.py): Helper utilities for preprocessing and evaluation
  * [minhashing.py](functions/minhashing.py): Functions for generating and using MinHash signatures
  * [tools.py](functions/tools.py): Helper functions for calculating similaritiesand matrix operations
* [.gitignore](.gitignore): File to specify files and directories ignored by Git
* [README.md](README.md): Project documentation
* LICENSE: License file for the project

---

## Technologies Used

### Libraries and Frameworks:
- **Apache Spark**: Used for distributed data processing and handling large-scale datasets efficiently.
- **Numpy**: For efficient numerical computations and array manipulations.  
- **Pandas**: For handling and preprocessing tabular data.  
- **Scikit-learn**: For clustering algorithms and evaluation metrics.  
- **Matplotlib**: For plotting and visualizing data.  
- **Plotly**: For interactive display utilities, used in animations.
- **nltk**: Used for text preprocessing.
- **yellowbrick.cluster (KElbowVisualizer)**: Used for determining the optimal number of clusters using the elbow method.
- **gensim.models.Word2Vec**: Used to represent text data as dense vector spaces for clustering or recommendations.  

---

## Project Overview

### **1. Recommendation System with LSH**
This system utilizes **Locality-Sensitive Hashing (LSH)** to recommend movies based on user preferences:
- **Steps:**  
  - **Data Preprocessing:** Cleaned and structured MovieLens data for effective use.  
  - **MinHash Implementation:** Compressed user data into concise signatures to approximate similarity measures.  
  - **LSH Application:** Divided user signatures into bands to efficiently identify similar users.  
  - **Recommendations:** Suggested movies by aggregating preferences from users deemed similar based on common ratings and their top-rated movies.

---

### **2. Clustering Movies**
We clustered movies based on their attributes and evaluated the clusters to understand group patterns:
- **Steps:**  
  - **Feature Engineering:** Extracted meaningful features, such as genres, average ratings, and tags.  
  - **Clustering Techniques:** Implemented and compared K-means, K-means++, and hierarchical clustering.  
  - **Evaluation:** Assessed cluster quality using metrics like silhouette score and Davies-Bouldin index.  
  - **Visualization:** Generated 2D plots to display clustering results and transitions during algorithm iterations.

---

### **3. Bonus Question**
We visualized clustering progression:
- **Steps:**  
  - **2D Animation:** Animated clustering iterations, showing how clusters formed and stabilized over time.  

---

### **4. Algorithmic Question**
Analyzed a strategic game where players maximize scores by selecting numbers from the sequence's ends:
- **Steps:**  
  - **Problem Description:** Two players, Arya and Mario, alternately pick numbers from either end of a sequence to maximize their scores.  
  - **Dynamic Programming:** Implemented a DP-based solution to compute optimal strategies for both players.  
  - **LLM Optimization:** Used insights from an LLM to refine and compare recursive and DP approaches.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
