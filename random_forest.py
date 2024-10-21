pip install scikit-learn
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Sample dataset (You can replace this with your dataset)
data = {
    'Age': [25, 45, 35, 50, 23, 51, 36, 40, 45, 22],
    'Income': [50000, 64000, 58000, 60000, 52000, 68000, 54000, 61000, 62000, 59000],
    'Student': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Credit_rating': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    'Buys_computer': [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Age', 'Income', 'Student', 'Credit_rating']]  # Features
y = df['Buys_computer']  # Target variable

# Split dataset into training set and test set (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)  # 100 trees in the forest

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Display feature importance
importances = clf.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f'Feature: {feature}, Importance: {importance}')
