import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from hyperopt import fmin, hp, tpe, Trials
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score

# Specify the path to your CSV file
file_path = r'C:\Users\flori\PycharmProjects\SCTP\preprocessed_file_cleaned.csv'

# Open the file in binary mode and read a chunk of data for analysis
df = pd.read_csv(file_path, delimiter=',')

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

space = {
    'num_leaves': hp.quniform('num_leaves', 10, 200, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'max_depth': hp.quniform('max_depth', 3, 20, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'min_child_samples': hp.quniform('min_child_samples', 50, 1000, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}


def objective(params):
    clf = lgb.LGBMClassifier(
        num_leaves=int(params['num_leaves']),
        max_depth=int(params['max_depth']),
        min_child_samples=int(params['min_child_samples']),
        n_estimators=1000,
        random_state=42
    )
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return -score


#trials = Trials()
#best = fmin(fn=objective,
 #           space=space,
 #           algo=tpe.suggest,
 #           max_evals=150,
 #           trials=trials,
 #           verbose=True)

#print("Best hyperparameters:", best)

params = {
    'num_leaves': 37,
    'max_depth': 4,
    'min_child_samples': 354,
}

model = lgb.LGBMClassifier(num_leaves=37, max_depth=4, min_child_samples=354)

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

# Plot feature importance
plt.figure(figsize=(10, 6))
# lgb.plot_importance(model, max_num_features=10, importance_type='gain', precision=0, ylabel='Sum Of Information Gain')
#plt.show()

# Plot split value histogram
plt.figure(figsize=(30, 35))
lgb.plot_split_value_histogram(model, feature=8, bins='auto')
plt.xlabel('Feature Value')
plt.ylabel('Count')
plt.title('Split Value Histogram for Blood Pressure Levels')
plt.show()
