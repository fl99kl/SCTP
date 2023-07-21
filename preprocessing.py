import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import OneHotEncoder
from pyECLAT import ECLAT


def convert_days_to_years(days):
    years = days // 365.25  # Assuming a leap year every 4 years
    return years


def convert_age_col(data):
    data['age'] = data['age'].apply(convert_days_to_years)
    # Define the ranges and labels
    age_ranges = [(0, 46), (46, 57), (57, float('inf'))]
    age_labels = [0, 1, 2]
    # Map the age ranges to the labels
    for i, (start, end) in enumerate(age_ranges):
        df.loc[(df['age'] >= start) & (df['age'] < end), 'age'] = age_labels[i]

    # Convert the 'age_encoded' column to integers
    data['age'] = data['age'].astype(int)


def insert_bmi(data):
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    # Define the BMI ranges and labels
    bmi_ranges = [(0, 18.5), (18.5, 25), (25, 30), (30, 35), (35, 40), (40, float('inf'))]
    bmi_labels = [0, 1, 2, 3, 4, 5]

    # Map the age ranges to the labels
    for i, (start, end) in enumerate(bmi_ranges):
        df.loc[(df['bmi'] >= start) & (df['bmi'] < end), 'bmi'] = bmi_labels[i]

    # Convert the 'bmi_encoded' column to integers
    data['bmi'] = data['bmi'].astype(int)
    data.drop(['height'], axis=1, inplace=True)
    data.drop(['weight'], axis=1, inplace=True)


# ap hi = systolic   ap_lo = diastolic
def process_blood_pressure(data):
    # Define the blood pressure criteria
    normal_range = np.logical_and(data['ap_lo'] < 80, data['ap_hi'] < 120)
    elevated_range = np.logical_and(data['ap_lo'] < 80, np.logical_and(120 <= data['ap_hi'], data['ap_hi'] <= 129))
    high_range_1 = np.logical_or(np.logical_and(80 <= data['ap_lo'], data['ap_lo'] <= 89), np.logical_and(130 <= data['ap_hi'], data['ap_hi'] <= 139))
    high_range_2 = np.logical_or(data['ap_lo'] >= 90, data['ap_hi'] >= 140)
    high_range_3 = np.logical_or(data['ap_lo'] >= 120, data['ap_hi'] >= 180)

    # Create a new column to indicate the blood pressure level and encode the values
    data['bp_level'] = ''
    data.loc[normal_range, 'bp_level'] = 0
    data.loc[elevated_range, 'bp_level'] = 1
    data.loc[high_range_1, 'bp_level'] = 2
    data.loc[high_range_2, 'bp_level'] = 3
    data.loc[high_range_3, 'bp_level'] = 4

    # Convert the 'bp_level' column to integers
    data['bp_level'] = data['bp_level'].astype(int)
    data.drop(['ap_hi'], axis=1, inplace=True)
    data.drop(['ap_lo'], axis=1, inplace=True)


def read_csv(file_path_string):
    data = pd.read_csv(file_path_string, delimiter=';')
    data.drop('id', inplace=True, axis=1)
    return data


# Specify the path to your CSV file
df = read_csv('C:/Users/flori/OneDrive/Desktop/cardio_train_cleaned.csv')
header = ''

# Do the preprocessing
convert_age_col(df)
insert_bmi(df)
process_blood_pressure(df)
# Specify name and path for DT data
df.to_csv('preprocessed_file_cleaned_new_bins.csv', index=False)

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the data to one-hot encode it
encoded_data = encoder.fit_transform(df)
# Create a DataFrame from the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns))
# Specify Name and path for the ARM data
encoded_df.to_csv('preprocessed_file_arm_cleaned_new_bins.csv', index=False)

encoded_df = encoded_df.apply(lambda row: encoded_df.columns[row.astype(bool)].tolist(), axis=1).tolist()

data = pd.DataFrame(encoded_df)

eclat_instance = ECLAT(data=data, verbose=True) #verbose=True to see the loading bar
print(eclat_instance.uniq_)    #a list with all the names of the different items



