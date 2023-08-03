import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time

# Specify the path to your CSV file
file_path = r'C:\Users\flori\PycharmProjects\SCTP\preprocessed_file_cleaned.csv'

# Open the file in binary mode and read a chunk of data for analysis
df = pd.read_csv(file_path, delimiter=',')

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

model = DecisionTreeClassifier(criterion='gini', max_depth=5, max_leaf_nodes=10, random_state=42, min_samples_split=155, min_impurity_decrease=7.384936101805374e-05)

start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(end - start)

start = time.time()
y_pred = model.predict(X_test)
end = time.time()
print(end - start)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(accuracy)
print(recall)
print(precision)

feature_names = ['age', 'gender', 'smoke', 'alco', 'active', 'cholesterol', 'gluc', 'bmi', 'bp_level']  # Replace with your actual feature names
importances = model.feature_importances_
feature_names = X.columns.tolist()
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feature_names, rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

# Plot the decision tree
plt.figure(figsize=(20, 20))
plot_tree(model,
          label='None',
          impurity=False,
          filled=True,
          rounded=True,
          proportion=True,
          feature_names=feature_names,
          fontsize=7,
          class_names=['noCardio', 'cardio'])
plt.show()
