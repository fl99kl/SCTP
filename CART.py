import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Specify the path to your CSV file
file_path = 'preprocessed_file_uncleaned.csv'

# Open the file in binary mode and read a chunk of data for analysis
df = pd.read_csv(file_path, delimiter=',')

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Create DT instance
model = DecisionTreeClassifier(criterion='gini', random_state=42, min_samples_split=155, min_impurity_decrease=7.384936101805374e-05)
# Train model
model.fit(X_train, y_train)
# predict the test set
y_pred = model.predict(X_test)
# show accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

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
plot_tree(model, filled=True, rounded=True, feature_names=feature_names, class_names=['noCardio', 'cardio'])
plt.show()
