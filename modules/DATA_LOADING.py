# Importing Necessary Libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from tabulate import tabulate

# Loading and Initial Analysis of the Fetal Health Dataset
data = pd.read_csv('C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/data.csv')

# %%
data.head(10)
data.shape
column_names = data.columns
column_data = pd.DataFrame(column_names, columns=['Column Names'])
column_data.index += 1
print(tabulate(column_data, headers='keys', tablefmt='fancy_grid', showindex=True))

data_types = data.dtypes
non_null_counts = data.count()
total_entries = [data.shape[0]] * len(data.columns)

info_data = pd.DataFrame({
    'Index': range(1, len(data.columns) + 1),
    'Feature': data.columns,
    'DataType': data_types,
    'Non-Null Count': non_null_counts,
    'Total Entries': total_entries
})

print(tabulate(info_data, headers='keys', tablefmt='fancy_grid', showindex='True'))


# Pre-processing / Data Cleaning
# Calculate summary statistics for each column
summary_stats = data.describe(include='all').T

# Add additional summary statistics
summary_stats['null_count'] = data.isnull().sum()
summary_stats['not_null_count'] = data.notnull().sum()
summary_stats['range'] = data.max() - data.min()

# Rename the columns for better readability
summary_stats.rename(columns={
    'count': 'Count',
    'mean': 'MeanValue',
    'std': 'StandardDeviation',
    'min': 'MinimumValue',
    '25%': '25thPercentile',
    '50%': 'Median',
    '75%': '75thPercentile',
    'max': 'MaximumValue',
    'null_count': 'NullValuesCount',
    'not_null_count': 'NotNullValuesCount',
    'range': 'Range'
}, inplace=True)
summary_stats.index.name = 'Feature'

summary_stats = summary_stats.round(2)

print(tabulate(summary_stats, headers='keys', tablefmt='fancy_grid'))

# Check values outside the expected range for numerical columns
numerical_columns = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']
found_out_of_range = False
for column in numerical_columns:
    value_min = data[column].min()
    value_max = data[column].max()
    out_of_range = data[(data[column] < value_min) | (data[column] > value_max)]
    if len(out_of_range) > 0:
        print(f"Potential values outside the expected range in column '{column}':")
        print(out_of_range)
        print()
        found_out_of_range = True

if not found_out_of_range:
    print("No values found outside the expected range in numerical columns.")

# Check for typos or inconsistent capitalization in categorical columns
categorical_columns = ['fetal_health']
found_inconsistencies = False
for column in categorical_columns:
    unique_values = data[column].astype(str).unique()
    inconsistent_values = [value for value in unique_values if value.lower() != value]
    if len(inconsistent_values) > 0:
        print(f"Potential inconsistencies in column '{column}':")
        print(inconsistent_values)
        print()
        found_inconsistencies = True

if not found_inconsistencies:
    print("No inconsistencies found in categorical columns.")

# Find constant columns
constant_columns = []
for column in data.columns:
    if data[column].nunique() == 1:
        constant_columns.append(column)

# Remove constant columns from the dataset
data = data.drop(constant_columns, axis=1)

# Print the list of constant columns
print("Constant Columns:")
if len(constant_columns) > 0:
    print(constant_columns)
else:
    print("No constant columns found.")

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
missing_values_table = [[i+1, col, val] for i, (col, val) in enumerate(missing_values.items())] 
print(tabulate(missing_values_table, headers=['Index', 'Column Name', 'Missing Values'], tablefmt='fancy_grid'))

# Find columns with missing values
columns_with_missing = missing_values[missing_values > 0].index.tolist()

if len(columns_with_missing) > 0:
    # Imputation: Replace missing values with mean of the entire dataset
    imputer = SimpleImputer(strategy='mean')
    data[columns_with_missing] = imputer.fit_transform(data[columns_with_missing])
    print("\nMissing values have been imputed.")

else:
    print("\nNo missing values found in the dataset.")

# Check for missing values after handling
missing_values_after = data.isnull().sum()

if missing_values_after.sum() == 0:
    print("All missing values have been handled.")
else:
    print("\nMissing Values After Handling:")
    missing_values_after_table = [[i+1, col, val] for i, (col, val) in enumerate(missing_values_after.iteritems())]
    print(tabulate(missing_values_after_table, headers=['Index', 'Column Name', 'Missing Values After Handling'], tablefmt='fancy_grid'))

# Check values outside the expected range for numerical columns
numerical_columns = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']
found_out_of_range = False
for column in numerical_columns:
    value_min = data[column].min()
    value_max = data[column].max()
    out_of_range = data[(data[column] < value_min) | (data[column] > value_max)]
    if len(out_of_range) > 0:
        print(f"Potential values outside the expected range in column '{column}':")
        print(out_of_range)
        print()
        found_out_of_range = True

if not found_out_of_range:
    print("No values found outside the expected range in numerical columns.")

# Check for typos or inconsistent capitalization in categorical columns
categorical_columns = ['fetal_health']
found_inconsistencies = False
for column in categorical_columns:
    unique_values = data[column].astype(str).unique()
    inconsistent_values = [value for value in unique_values if value.lower() != value]
    if len(inconsistent_values) > 0:
        print(f"Potential inconsistencies in column '{column}':")
        print(inconsistent_values)
        print()
        found_inconsistencies = True

if not found_inconsistencies:
    print("No inconsistencies found in categorical columns.")

#Changing the data type of target column from float to int for classification
data['fetal_health'] = data['fetal_health'].astype(int)
data['fetal_health']

# Saving the processed data to a CSV file
processed_data_path = 'C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/processed_data.csv'
data.to_csv(processed_data_path, index=False)
print(f"Processed data saved to {processed_data_path}")

# Displaying the first few rows of the processed dataset
print("/nFirst few rows of the processed data:")
print(tabulate(data.head(), headers='keys', tablefmt='fancy_grid'))


