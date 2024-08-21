# Sentiment-Analysis-Twitter-Dataset-NLP-ASSIGNMENT-
This repository contains a complete end-to-end implementation of sentiment analysis on a large-scale Twitter dataset. The project focuses on predicting the sentiment of tweets, classified into three categories: negative (0), neutral (2), and positive (4). It uses NLP preprocessing techniques, vectorization, and various machine learning models to achieve this task.
# Sentiment Analysis on Twitter Dataset

## Overview

This project aims to perform sentiment analysis on a large-scale Twitter dataset. The dataset contains labeled tweets with sentiments classified into three categories:
- **0**: Negative
- **2**: Neutral
- **4**: Positive

The goal is to predict the sentiment of the tweets using various machine learning models after preprocessing the text data.

## Project Structure

- **data/**: Contains the original dataset file.
- **notebooks/**: Jupyter notebook with all the code implementations.
- **models/**: Saved models after training.
- **README.md**: Documentation file.
- **requirements.txt**: List of dependencies required to run the project.

## Preprocessing

Text data is preprocessed using the following steps:
1. **Tokenization and Lemmatization**: Tokenizing the text and converting words to their base forms.
2. **Data Cleansing**: Removing stopwords, URLs, and special characters.

## Vectorization

The preprocessed text is converted into numerical representations using:
- **CountVectorizer**
- **TFIDFVectorizer**

## Machine Learning Models

The following models are trained and evaluated:
1. **Logistic Regression**
2. **Support Vector Classifier (SVC)**
3. **Random Forest**

Each model is trained and tested using both `CountVectorizer` and `TFIDFVectorizer`.

## Results

The performance of each model is evaluated using classification reports and confusion matrices. Heatmaps are provided for visual comparison of model accuracy.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/RiyaGupta122/Sentiment-Analysis-Twitter-Dataset-NLP-ASSIGNMENT-
    ```
   
2. **Navigate to the project directory**:
    ```bash
    cd Sentiment-Analysis-Twitter-Dataset
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the notebook**:
    Open the `Sentiment_Analysis.ipynb` notebook in Jupyter and run the cells to preprocess data, train models, and evaluate results.

## Dependencies

- Python 3.7+
- Pandas
- Numpy
- NLTK
- SpaCy
- Scikit-learn
- Matplotlib
- Seaborn

Install the required packages using:
## bash
pip install -r requirements.txt


### Summary of Content:

- **Overview**: Introduces the project, dataset, and objectives.
- **Project Structure**: Describes the folder structure and purpose of each directory/file.
- **Preprocessing**: Details the text preprocessing steps.
- **Vectorization**: Explains how text is converted into numerical data.
- **Machine Learning Models**: Lists the models used and their purpose.
- **Results**: Mentions how results are evaluated.
- **How to Use**: Provides instructions on how to clone the repository, install dependencies, and run the code.
- **Dependencies**: Lists the Python packages required.
- **Conclusion**: Summarizes the project outcome.

## Learning Outcome
### Natural Language Processing (NLP) Techniques:

1) Understand and apply text preprocessing techniques including tokenization, lemmatization, and stopword removal.
Gain experience with cleaning and preparing real-world textual data for machine learning tasks.

### Vectorization Methods:

1) Learned how to convert textual data into numerical representations using CountVectorizer and TFIDFVectorizer.
Understand the differences between these vectorization techniques and how they impact model performance.

### Machine Learning Model Training:

1) Gain hands-on experience with training and tuning machine learning models, specifically Logistic Regression, Support Vector Classifier (SVC), and Random Forest, on textual data.
Learn to evaluate model performance using classification reports and confusion matrices.

### Model Evaluation and Comparison:

1) Develop skills in interpreting classification metrics such as precision, recall, F1-score, and accuracy.
Compare the performance of different models and vectorization methods using visualizations like heatmaps of confusion matrices.

### Python Programming:

1) Enhance your ability to use Python for data science, including working with popular libraries such as Pandas, Scikit-learn, NLTK, and SpaCy.
Gain experience in structuring a complete machine learning project in Python, including data loading, preprocessing, model training, and evaluation.


Riya Gupta 
AIML A3
22070126089




