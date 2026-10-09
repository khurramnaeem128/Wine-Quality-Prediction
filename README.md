# Wine Quality Prediction Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to predict wine quality based on its physicochemical properties. The model analyzes numerical features of wine and predicts its quality category using the available dataset.

The project demonstrates a complete beginner-level machine learning workflow, including data preprocessing, feature standardization, model training, and performance evaluation.

## Objectives

* Analyze wine quality data.
* Understand the relationship between wine properties and quality.
* Prepare data for machine learning.
* Train a classification model to predict wine quality.
* Evaluate model performance using accuracy.
* Understand the challenges of predicting quality from numerical features.

## Dataset

The project uses the **Wine Quality Red Dataset**, which contains measurements of red wine samples.

Features include:

* Fixed acidity
* Volatile acidity
* Citric acid
* Residual sugar
* Chlorides
* Free sulfur dioxide
* Total sulfur dioxide
* Density
* pH
* Sulphates
* Alcohol

**Target:** Wine quality score.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib (if used for visualization)
* Scikit-learn

## Machine Learning Workflow

### 1. Data Loading

Load the wine quality dataset and inspect its rows, columns, data types, and statistical information.

### 2. Data Preprocessing

Separate the input features from the target quality column. Prepare the data for model training and check for potential data quality issues.

### 3. Feature Standardization

Apply feature standardization so numerical features are on comparable scales. This is particularly useful for models sensitive to feature magnitude.

### 4. Train-Test Split

Divide the dataset into training and testing sets to evaluate how well the model performs on unseen data.

### 5. Model Training

Train a machine learning classification model using the training data.

### 6. Model Evaluation

Evaluate the model using classification accuracy and compare predicted quality labels with actual labels.

## Results

The project achieved the following accuracy scores:

| Model                             | Accuracy |
| --------------------------------- | -------: |
| Standardized classification model |   58.82% |
| Decision Tree Classifier          |   49.26% |

These results indicate that predicting wine quality from physicochemical measurements is a challenging classification task. Performance can depend on the model, feature representation, class distribution, and evaluation strategy.

## Key Learnings

* Data preprocessing for machine learning.
* Separating features and target variables.
* Feature standardization.
* Training and testing classification models.
* Evaluating models using accuracy.
* Understanding the limitations of classification performance.

## Future Improvements

* Evaluate a Random Forest Classifier.
* Use a confusion matrix and classification report.
* Examine class imbalance in the target variable.
* Experiment with different classification algorithms.
* Tune model hyperparameters.
* Explore alternative target-label strategies, such as grouping quality scores into broader categories.

## Applications

* Wine quality analysis.
* Food and beverage quality research.
* Classification of products using measurable properties.
* Educational practice in machine learning.

## Conclusion

This project demonstrates how machine learning can be applied to wine quality classification using physicochemical data. The results also highlight the importance of data preprocessing, model selection, and appropriate evaluation when building predictive systems.
