# Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from tabulate import tabulate

# Importing Necessary Libraries
data = pd.read_csv("C:/Users/Vidhi/Documents/PROJECTS/FETAL HEALTH PROJECT/data/processed_data.csv")

# Exploratory Data Analysis (EDA)
# Calculate skewness for each feature
skewness = data.skew()
skewness = skewness.round(2)
skewness_data = pd.DataFrame(skewness, columns=['Skewness']).reset_index()
skewness_data.index += 1
skewness_data.rename(columns={'index': 'Feature'}, inplace=True)
print("Skewness:")
print(tabulate(skewness_data, headers='keys', tablefmt='fancy_grid'))

# Calculate kurtosis for each column
kurtosis_values = data.apply(kurtosis)
kurtosis_data = pd.DataFrame(kurtosis_values, columns=['Kurtosis']).reset_index()
kurtosis_data.columns = ['Feature', 'Kurtosis']
kurtosis_data.index += 1
print("Kurtosis Values:")
print(tabulate(kurtosis_data, headers='keys', tablefmt='fancy_grid', showindex=True))

# Count the number of unique values in each categorical column
categorical_columns = ['fetal_health']
for column in categorical_columns:
    unique_values = data[column].nunique()
    print(f"Number of unique values in {column}: {unique_values}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Calculating the Percentage Distribution of the Target Variable
fetal_health_counts = data['fetal_health'].value_counts()
fetal_health_percentage = fetal_health_counts / fetal_health_counts.sum() * 100
fetal_health_data = pd.DataFrame(fetal_health_percentage).reset_index()
fetal_health_data.columns = ['index', 'percentage']

# Map labels to the fetal health classes
labels = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
fetal_health_data['index'] = fetal_health_data['index'].map(labels)
fetal_health_data.columns = ['Fetal Health Class', 'Percentage']

# Print the percentage distribution in a tabular format
print("Percentage Distribution of the Fetal Health Class:")
print(tabulate(fetal_health_data, headers='keys', tablefmt='fancy_grid', showindex=False, numalign="right"))

# Visualization: Plotting the Percentage Distribution of Fetal Health
plt.figure(figsize=(8, 6))
sns.barplot(x='Fetal Health Class', y='Percentage', data=fetal_health_data, palette='muted')

# Add value labels on top of bars
for index, row in fetal_health_data.iterrows():
    plt.text(index, row['Percentage'] + 0.5, f"{row['Percentage']:.1f}%", ha='center')

# Set plot title and labels
plt.title("Percentage Distribution of Fetal Health Classes")
plt.ylabel("Percentage (%)")
plt.xlabel("Fetal Health Class")

# Display the plot
plt.tight_layout()
plt.show()

# Set up the figure and axes for the plots
fig, ax = plt.subplots(7, 3, figsize=(20, 30))
ax = ax.ravel()
# Get the feature columns by dropping the target column
df_feature = data.drop(["fetal_health"], axis=1)
# For each feature column, plot a histogram
for i, col in enumerate(df_feature.columns):
    sns.histplot(data, x=col, hue='fetal_health', kde=True, bins=80, ax=ax[i],  palette='muted', alpha=0.4)
    ax[i].set_title(col)
    ax[i].grid()

plt.tight_layout()
plt.show()

# Plot boxplots for each feature
plt.figure(figsize=(20, 20))
for i, col in enumerate(data.columns):
    plt.subplot(5, 5, i+1)
    sns.boxplot(data[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Calculate the correlation coefficients
correlations = data.corr()['fetal_health']
del correlations['fetal_health']

# Sort the correlations
sorted_correlations = correlations.sort_values(ascending=False)

# Convert series to DataFrame for tabulate
correlations_data = pd.DataFrame(sorted_correlations).reset_index()
correlations_data.columns = ['Feature', 'Correlation with Fetal Health']
correlations_data.index += 1

# Display the correlations in a tabular form
print("Correlation Coefficients with Fetal Health:")
print(tabulate(correlations_data, headers='keys', tablefmt='fancy_grid', showindex=True))

# Examining correlation matrix using heatmap
cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
cols = ["#17ffc4", "#ffdd00", "#00ffff", "#ff006d", "#adff02", "#8f00ff"]
corrmat= data.corr()
f, ax = plt.subplots(figsize=(20,15))
sns.heatmap(corrmat, cmap=cols, annot=True)
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()

cols = ["#17ffc4", "#ffdd00", "#00ffff", "#ff006d", "#adff02", "#8f00ff"]

# Pearson's correlation matrix
correlation_matrix = data.corr()
plt.subplots(figsize=(25,1))
sns.heatmap(correlation_matrix.sort_values(by=["fetal_health"], ascending=False).head(1),
            annot=True, cmap=sns.color_palette(cols))
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()

# Correlation with Target Variable
plt.figure(figsize=(10, 6),dpi=100)
data.corr()['fetal_health'].sort_values()[:-1].plot(kind='barh', color='#fb8500')
plt.title('Correlation of Features with Target Column (Fetal Health)')
plt.xlabel('Correlation')
plt.ylabel('Features')
plt.show()


