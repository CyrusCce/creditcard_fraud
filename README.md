# Credit Card Fraud Detection Project
Credit Card Fraud Detection Using Machine Learning and Deep Learning Techniques

## Overview
This project is part of a mini-project for SC1015 (Introduction to Data Science and Artificial Intelligence), focusing on detecting credit card fraud using statistical modeling, machine learning, and deep learning techniques. The project aims to identify fraudulent transactions and mitigate losses by leveraging advanced analytical methods.

## Problem Definition
The goal is to help banks and financial institutions recognize fraudulent credit card transactions to reduce financial losses and customer inconvenience. This is increasingly important as credit card fraud continues to evolve with technological advancements.

## Dataset
The dataset comprises transactions from European credit cardholders in September 2013, featuring 284,807 transactions, of which 492 are fraudulent. This represents a highly imbalanced dataset with only 0.172% of transactions being fraudulent. The data features are the result of a PCA transformation to protect user identities, with features V1 to V28 as principal components, and 'Time' and 'Amount' as non-transformed features. The target variable 'Class' indicates whether a transaction is fraudulent.

Handling Imbalanced Data
Given the highly imbalanced dataset, the project uses the following techniques to address class imbalance:
- Synthetic Minority Over-sampling Technique (SMOTE): Increases the minority class by creating synthetic samples based on k-nearest neighbors.
- Random UnderSampling Technique (RUS): Randomly removes samples from the majority class to achieve a desired balance.
Combining SMOTE with RUS yielded optimal results, achieving an ROC score of 1.

## Installation Instructions
1. **Clone the repository** to your local machine.
2. **Ensure Python 3.7 or later is installed.**
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install tensorflow
   pip install scikit-learn
   ```
4. **Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the appropriate directory.**

## Usage
1. Import the `creditcard.csv` file into a Jupyter notebook.
2. Execute all cells sequentially to perform data preprocessing, model training, and evaluation.
3. Explore the data and results through the provided Jupyter Notebooks.

## Configuration
Ensure your Python and Jupyter environments are correctly configured with all required dependencies installed.

## Models Used
- **Decision Tree**
- **Random Forest**
- **Logistic Regression**
- **Convolutional Neural Network (CNN)**

## Jupyter Notebooks Overview
[Data Extraction & Exploratory Analysis]
It includes:
- Data extraction from various sources to create datasets.
- Grouping and preprocessing of data, including PCA transformation.
- Exploratory analysis to understand variable distribution and correlations.
- Visualization techniques such as histograms, scatter plots, and correlation matrices.

[Machine Learning - Model Building]
This notebook explores machine learning models for fraud detection. It includes:
- Implementation of decision tree, random forest, and logistic regression models.
- Hyperparameter tuning to optimize model performance.
- Evaluation of models using metrics like accuracy, ROC AUC, and confusion matrices.
- Handling of imbalanced data using techniques like SMOTE and RUS.
- Analysis of model performance and comparison of results.

[Notebook #3: Deep Learning - Convolutional Neural Network (CNN)]
This notebook focuses on using deep learning techniques, specifically Convolutional Neural Networks (CNNs), to detect credit card fraud. It covers:
- Designing and training a CNN model for fraud detection.
- Architecture of the CNN model, including convolutional and dense layers.
- Evaluation of the CNN model using metrics like accuracy, loss, and ROC AUC.
- Comparison of CNN performance with traditional machine learning models.
- Analysis of results and recommendations for model deployment.

## Statistical Analysis and Recommendations
Although both logistic regression and CNN models achieved high test accuracy, the CNN model demonstrated a higher accuracy score, indicating its superior capability in complex pattern recognition tasks like credit card fraud detection. This leads to the following recommendations:
- SMOTE followed by RUS: To address class imbalance, using a combination of SMOTE and RUS is recommended, leading to improved model performance.
- Convolutional Neural Network (CNN) 1D sequence: This structure, with multiple convolutional and dense layers, offers high accuracy rates without compromising on quality and integrity.
- To further improve performance, consider adding a MaxPooling1D layer to reduce spatial dimensions while retaining key features. This may lead to faster training times and reduced overfitting.

## Contributing
Contributions are welcome. To contribute to this project:
- Fork the repository and create a new branch.
- Implement your changes and submit a pull request.
- Provide a clear description of your changes and the purpose of the pull request.
- Follow the project's code of conduct and best practices for contributing.

## Credits
This project was developed by Group 9 for SC1015. 

## License
This project is licensed under the MIT License - see the LICENSE file for details.
