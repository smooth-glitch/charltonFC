# Data Scientist Intern Entry Task:

Welcome to the Data Scientist Intern Task repository! This project demonstrates data analysis and visualization using Python. Below you'll find an overview of the project, the tech stack used, and instructions on how to set it up and run it on your local system.

## Project Overview

The goal of this project was to analyze a dataset containing player statistics. Key objectives included:

- **Data Cleaning**: Handling missing values and standardizing column names.
- **Data Transformation**: Normalizing data and creating new metrics.
- **Analysis**: Identifying top performers based on a combined score of multiple metrics.
- **Visualization**: Generating visualizations to better understand the data.
- **Machine Learning**: Using AI algorithms like Random Forest and K-Means to generate scores based on player performance.


## 💻 Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![OpenSea](https://img.shields.io/badge/OpenSea-%232081E2.svg?style=for-the-badge&logo=opensea&logoColor=white)

## Machine Learning Algorithms
1. Random Forest
Random Forest is a supervised learning algorithm that is used for both classification and regression tasks. It works by constructing multiple decision trees at training time and outputting either the average prediction (for regression tasks) or the majority vote (for classification tasks) of the individual trees.

Key Characteristics:
It reduces the risk of overfitting by averaging the results of multiple trees.
Randomly selects data points and features to train each tree, ensuring variety in the trees and reducing correlation between them.
It can estimate the importance of each feature in making predictions.
In this project, a Random Forest Regressor is used to generate a new AI-based score based on two columns (col1 and col2). The function:

Trains the Random Forest on the average of the two columns.
Predicts an AI score that combines the two features in a data-driven manner.
The function also uses Random Forest to train a model that determines optimal weights for various performance metrics, which helps to predict the final score for a player.

2. K-Means Clustering
K-Means is an unsupervised learning algorithm that is used for clustering. The goal is to partition the data into K clusters based on the similarity of the data points, such that data points within the same cluster are more similar to each other than to those in other clusters.

Key Characteristics:
Works by iteratively assigning data points to the nearest centroid and then recalculating the centroids of the clusters.
It continues the process until the centroids stop changing, meaning the clusters are stable.
Commonly used for segmentation, anomaly detection, and pattern recognition.
In this project, K-Means is used to group players into 3 clusters based on two performance metrics (col1 and col2). Each player is assigned a cluster label (ai_score), representing their similarity to other players based on their statistics.

3. AI-Based Score Generation
The create_ai_based_score function in this project allows for generating scores using either Random Forest or K-Means:

Random Forest: Combines col1 and col2 using a trained model to generate a continuous ai_score.
K-Means: Groups players into clusters based on col1 and col2, and assigns a categorical ai_score corresponding to the cluster.
This flexible approach helps analyze player performance using both supervised and unsupervised learning techniques.

## Timeframe

This project was completed over a period of 6 days, including the following phases:

- **Day 1**: Data exploration, cleaning, and transformation.
- **Day 2**: Data analysis, and visualization using matplotlib.
- **Day 3**: visualization enhancement using Tableau -> from matplotlib to tableau.
- **Day 4**: thorough documentation and creation of a detailed presentation.
- **Day 5**: code review and potential enhancements for the processing of the dataset.
- **Day 6**: Submission and presentation to the firm.

## Project Structure

The project folder structure is available here:

https://freeimage.host/i/dv4gT12

## Setup Instructions

To set up this project on your local system, follow these steps:

1. **Clone the Repository**:

   ```bash
   https://github.com/smooth-glitch/charltonFC.git
   cd charltonFC
   
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv

3. **Activate the Virtual Environment**:
   
	  ```bash
   		cd venv
   		./scripts/activate
   
5. **Install Dependencies**:
   ```bash
      pip install -r requirements.txt

6. **Run the Main Script**:
   ```bash
      python main.py

Ensure that the dataset (DataScientistInternTask.csv) is located in the data/ directory.

### Dataset
   Due to the large size of the dataset, it is not included in this repository. You can download it from Google Drive. Place the file in the data/ directory to run the analysis.
   this is the G-drive link for the dataset : https://drive.google.com/file/d/1kb0ivHsLaKdVb1jZkwyuIYYzQewu0dmn/view?usp=drive_link.
   
   NOTE: YOU WILL NEED ACCESS PERMISSIONS FOR THIS.

### Tableau dashboard glimpse:
![Diagram](https://github.com/smooth-glitch/charltonFC/blob/main/sample_output.png)

### 💻Presentation:
[VIEW THE PRESENTATION](https://github.com/smooth-glitch/charltonFC/raw/main/charltonFC.pdf)

### Contribution

   If you'd like to contribute to this project, please fork the repository and create a pull request with your changes. Make sure to include a description of the changes and any relevant details.

### License
    
   This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
    
   For any questions or feedback, feel free to reach out to arjunsridhar445@gmail.com.
   
### 💰 You can help me by Donating
  [![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/smoothglitch) 

  
<!-- Proudly created with GPRM ( https://gprm.itsvg.in ) -->
