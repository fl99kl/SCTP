import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import randint as sp_randint

# Specify the path to your CSV file
file_path = 'C:/Users/flori/PycharmProjects/pythonProject/preprocessed_file_uncleaned.csv'

# Open the file in binary mode and read a chunk of data for analysis
df = pd.read_csv(file_path, delimiter=',')

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Plot feature importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10)
plt.show()

# Plot split value histogram
feature_index = 0  # Specify the index of the feature you want to plot
plt.figure(figsize=(30, 35))
lgb.plot_split_value_histogram(model, feature=feature_index, bins='auto')
plt.xlabel('Feature Value')
plt.ylabel('Count')
plt.title('Split Value Histogram for Age')
plt.show()
