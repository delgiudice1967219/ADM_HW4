# ADM - Homework 4: Movie Recommendation System, Group #10

This GitHub repository contains the implementation of the fourth homework of the **Algorithmic Methods of Data Mining** course for the master's degree in Data Science at Sapienza (2024-2025). This homework was completed by Group #10. The details of the assignement are specified here:  https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_4

**Team Members:**
* Xavier Del Giudice, 1967219, delgiudice.1967219@studenti.uniroma1.it
* Valeria Avino, 1905974, avino.1905974@studenti.uniroma1.it
* Mattia Mungo, 1883175, Mungo.1883175@studenti.uniroma1.it
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
  * [tools.py](functions/tools.py): Helper functions for find similarity
* [.gitignore](.gitignore): File to specify files and directories ignored by Git
* [README.md](README.md): Project documentation
* LICENSE: License file for the project

---

## Project Overview

### **1. Recommendation System with LSH**
We implemented a recommendation system based on the **Locality-Sensitive Hashing (LSH)** algorithm, using the MovieLens dataset to identify similar users based on their movie preferences. Key steps include:
- Preprocessing data from the MovieLens dataset.
- Implementing MinHash from scratch to create compact user signatures.
- Applying LSH to group similar users.
- Recommending movies based on user similarity, considering common ratings and top-rated movies of similar users.

### **2. Grouping Movies Together**
Using clustering techniques, we grouped movies based on extracted features. Key highlights include:
- **Feature Engineering:** Selecting and creating features such as genres, average ratings, and user-assigned tags.
- **Clustering Algorithms:** Implemented K-means, K-means++, and a third algorithm recommended by an LLM.
- **Evaluation Metrics:** Assessed cluster quality using metrics like silhouette score, Davies-Bouldin index, and intra-cluster distance.

### **3. Bonus Question**
We visualized the progression of clustering over iterations. Using two key variables, we plotted 2D graphs to display how clusters formed and refined.

### **4. Algorithmic Question**
This section involved solving a strategic game where two players, Arya and Mario, compete to maximize their score by picking numbers from the ends of a sequence. Our solution:
- Used recursive and dynamic programming approaches to ensure optimal strategies for both players.
- Compared the efficiency of different implementations and leveraged LLMs to suggest optimizations.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
