# Hate Speech Detection: An NLP Pipeline for Text Classification

![Hate Speech Detection](https://img.shields.io/badge/Hate%20Speech%20Detection-v1.0-blue)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data Exploration](#data-exploration)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction Script](#prediction-script)
- [Contributing](#contributing)
- [License](#license)
- [Links](#links)

## Overview
This repository contains a complete Natural Language Processing (NLP) pipeline designed to detect hate speech, offensive language, and neutral content. The project utilizes TF-IDF (Term Frequency-Inverse Document Frequency) along with various machine learning algorithms to classify text effectively. The repository includes essential components such as exploratory data analysis (EDA), data preprocessing, model training, evaluation, and a reusable Python script for making predictions.

## Features
- Comprehensive NLP pipeline for hate speech detection
- Utilizes TF-IDF for feature extraction
- Implements various machine learning models, including Random Forest
- Includes exploratory data analysis (EDA) to understand the dataset
- Offers a Python script for easy predictions
- Well-documented Jupyter notebooks for learning and reference

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Installation
To get started with this project, clone the repository and install the required packages.

```bash
git clone https://github.com/emojipasta/Hate-Speech-Detection.git
cd Hate-Speech-Detection
pip install -r requirements.txt
```

## Usage
After installation, you can explore the Jupyter notebooks for a detailed understanding of the data processing and model training steps. 

To run the prediction script, use the following command:

```bash
python predict.py --input "Your text here"
```

This will output whether the input text is hate speech, offensive, or neutral.

## Data Exploration
The dataset used for this project is critical for training the models. The EDA section of the Jupyter notebook provides insights into the distribution of classes, common words, and other useful statistics.

### Key Insights
- Distribution of hate speech vs. neutral content
- Common words in each category
- Visualization of class distribution

## Model Training
The project implements several machine learning models to classify the text data. The primary model used is the Random Forest classifier, known for its robustness and accuracy.

### Training Process
1. **Data Preprocessing**: Cleaning and preparing the data for training.
2. **Feature Extraction**: Using TF-IDF to convert text into numerical format.
3. **Model Selection**: Choosing the best-performing model based on evaluation metrics.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

The evaluation section provides detailed results for each model trained.

## Evaluation
After training the models, we evaluate their performance using a separate test dataset. The results are visualized to understand how well the models classify hate speech.

### Results
- Confusion matrix for each model
- ROC curves
- Comparison of model performance

## Prediction Script
The repository includes a reusable Python script for making predictions on new text. This script allows users to input text and receive immediate feedback on its classification.

To download the latest version of the script, visit the [Releases section](https://github.com/emojipasta/Hate-Speech-Detection/releases).

## Contributing
Contributions are welcome! If you have suggestions for improvements or want to add features, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Links
For more details and to download the latest release, visit the [Releases section](https://github.com/emojipasta/Hate-Speech-Detection/releases). 

![Hate Speech Detection](https://img.shields.io/badge/Visit%20Releases-blue)