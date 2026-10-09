
# WINE QUALITY PREDICTION - CLASSIFICATION

# 1. LOAD DATASET


import pandas as pd

df_original = pd.read_csv("winequality-red.csv")
df = df_original.copy()

# 2. UNDERSTAND THE DATA

# Display first 5 rows
print(df.head())

# Display rows and columns
print(df.shape)

# Display column information
df.info()

# Display statistical summary
print(df.describe())

# Check missing values
print(df.isnull().sum())


# 3. REMOVE DUPLICATE ROWS

df = df.drop_duplicates()

# Check remaining duplicate rows
print(df.duplicated().sum())


# 4. CHECK WINE QUALITY DISTRIBUTION

# Count each wine quality value
print(df["quality"].value_counts().sort_index())


# 5. VISUALIZE WINE QUALITY DISTRIBUTION

import matplotlib.pyplot as plt

quality_counts = df["quality"].value_counts().sort_index()

quality_counts.plot(kind="bar")

plt.xlabel("Wine Quality")
plt.ylabel("Number of Wines")
plt.title("Wine Quality Distribution")
plt.show()


# 6. VISUALIZE FEATURE DISTRIBUTIONS


# Create histograms for all numerical features
df.hist(figsize=(14, 12), bins=20)

plt.tight_layout(h_pad=2.5)
plt.show()


# 7. SEPARATE FEATURES AND TARGET

# X contains input features
X = df.drop("quality", axis=1)

# y contains target variable
y = df["quality"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print(X.head())
print(y.head())


# 8. TRAIN-TEST SPLIT


# Use 80% data for training and 20% for testing
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Split index:", split_index)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Display test data index range
print("First test row:", X_test.index[0])
print("Last test row:", X_test.index[-1])

# 9. FEATURE SCALING


from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


X_test_scaled = scaler.transform(X_test)

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)



# 10. LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train_scaled, y_train)

print("Model trained successfully")

# Make predictions
y_pred = model.predict(X_test_scaled)

print("Predicted values:")
print(y_pred[:10])



# 11. ACTUAL VS PREDICTED VALUES


comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(comparison.head(10))


# 12. LOGISTIC REGRESSION ACCURACY


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100)


# 13. LOGISTIC REGRESSION CONFUSION MATRIX


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# 14. LOGISTIC REGRESSION CLASSIFICATION REPORT


from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 15. DECISION TREE

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
print("Decision Tree trained successfully")


# 16. DECISION TREE ACCURACY

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("Decision Tree Accuracy:", tree_accuracy)
print("Decision Tree Accuracy Percentage:", tree_accuracy * 100)


# 17. DECISION TREE CONFUSION MATRIX

cm_tree = confusion_matrix(y_test, y_pred_tree)

print("Decision Tree Confusion Matrix:")
print(cm_tree)

# 18. DECISION TREE CLASSIFICATION REPORT


print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree, zero_division=0))

# 19. RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

# Make predictions
y_pred_forest = forest_model.predict(X_test)
print("Random Forest trained successfully")

# 20. RANDOM FOREST ACCURACY

forest_accuracy = accuracy_score(y_test, y_pred_forest)

print("Random Forest Accuracy:", forest_accuracy)
print("Random Forest Accuracy Percentage:", forest_accuracy * 100)


# 21. RANDOM FOREST CONFUSION MATRIX


cm_forest = confusion_matrix(y_test, y_pred_forest)

print("Random Forest Confusion Matrix:")
print(cm_forest)


# 22. RANDOM FOREST CLASSIFICATION REPORT

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_forest, zero_division=0))


# 23. INITIAL MODEL COMPARISON


print("\nModel Comparison")

print("Logistic Regression:", accuracy * 100, "%")
print("Decision Tree:", tree_accuracy * 100, "%")
print("Random Forest:", forest_accuracy * 100, "%")

# 24. GRID SEARCH


from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("GRID SEARCH COMPLETED")

# 25. BEST GRID SEARCH PARAMETERS


print("Best Parameters:")
print(grid_search.best_params_)

print("Best CV Accuracy:")
print(grid_search.best_score_)


# 26. TUNED RANDOM FOREST


# Create Random Forest using best parameters
best_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train tuned Random Forest
best_forest.fit(X_train, y_train)

# Make predictions
y_pred_best = best_forest.predict(X_test)
print("Tuned Random Forest trained successfully")



# 27. TUNED RANDOM FOREST ACCURACY


best_accuracy = accuracy_score(y_test, y_pred_best)

print("Tuned Random Forest Accuracy:", best_accuracy)
print("Tuned Random Forest Accuracy Percentage:", best_accuracy * 100)


# 28. TUNED RANDOM FOREST CONFUSION MATRIX

best_cm = confusion_matrix(y_test, y_pred_best)

print("Tuned Random Forest Confusion Matrix:")
print(best_cm)

# 29. TUNED RANDOM FOREST CLASSIFICATION REPORT


print("Tuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best, zero_division=0))



# 30. FINAL MODEL COMPARISON


model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "Accuracy": [
        accuracy,
        tree_accuracy,
        forest_accuracy,
        best_accuracy
    ]
})

# Convert accuracy into percentage
model_comparison["Accuracy Percentage"] = (
    model_comparison["Accuracy"] * 100
)

print("\nFinal Model Comparison:")
print(model_comparison)



# 31. PREDICT A SPECIFIC WINE

row_number = int(input("\nEnter original row number: "))

if row_number not in df_original.index:

    print("Invalid row number. Please enter a number from 0 to 1598.")

else:

    
    new_wine = df_original.drop("quality", axis=1).loc[[row_number]]

    
    actual_quality = df_original.loc[row_number, "quality"]

    
    prediction = best_forest.predict(new_wine)[0]

    print("\nWine Data:")
    print(new_wine)

    print("Actual Wine Quality:", actual_quality)
    print("Predicted Wine Quality:", prediction)

    
    if prediction == actual_quality:
        print("Prediction: Correct")
    else:
        print("Prediction: Wrong")


# 32. FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_forest.feature_importances_
})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# 33. FEATURE IMPORTANCE VISUALIZATION

plt.figure(figsize=(10, 6))


plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Wine Quality Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# 34. FINAL PROJECT RESULTS


print("\n" + "=" * 50)
print("           FINAL PROJECT RESULTS")
print("=" * 50)

print("Best Model        :", "Tuned Random Forest")
print("Best Accuracy     :", f"{best_accuracy * 100:.2f}%")
print(
    "Most Important Feature:",
    feature_importance.iloc[0]["Feature"]
)
