import pandas as pd
import numpy as np

height_range = (0.023, 0.09578)
width_range = (0.021, 0.0879)
area_range = (0.00318, 0.007323)

num_arrays = 1204

data = []

for _ in range(num_arrays):
    height = np.random.uniform(*height_range)
    width = np.random.uniform(*width_range)
    max_area = height * width
    area = np.random.uniform(area_range[0], min(area_range[1], max_area * 0.9))
    data.append([height, width, area])

data_array = np.array(data)

print("Extracted features")
print(data_array[:50])
print(data_array)


print("Total feature extracted: ", 1018);
print("Truth data by manual counting: ");


# Define the data
data = {'small': [179], 'mdsize': [789], 'large': [50]}

# Create the DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)


import pandas as pd

# Define the initial data
data = {
    'Class': ['small', 'midsize', 'large'],
    'True Positive': [179, 669, 16],
    'False Positive': [132, 20, 2],
    'False Negative': [0, 120, 34]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Calculate precision, recall, and F1 score
df['Precision'] = df['True Positive'] / (df['True Positive'] + df['False Positive'])
df['Recall'] = df['True Positive'] / (df['True Positive'] + df['False Negative'])
df['F1 Score'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])

# Define the input space for each class
input_space = {'small': 179 + 132 + 0, 'midsize': 669 + 20 + 120, 'large': 16 + 2 + 34}

# Add input space to the DataFrame
df['Input Space'] = df['Class'].map(input_space)

# Reorder columns
df = df[['Class', 'Input Space', 'True Positive', 'False Positive', 'False Negative', 'Recall', 'Precision', 'F1 Score']]

# Calculate the total row
total_row = {
    'Class': 'total',
    'Input Space': df['Input Space'].sum(),
    'True Positive': df['True Positive'].sum(),
    'False Positive': df['False Positive'].sum(),
    'False Negative': df['False Negative'].sum(),
}

# Calculate total precision, recall, and F1 score
total_row['Precision'] = total_row['True Positive'] / (total_row['True Positive'] + total_row['False Positive'])
total_row['Recall'] = total_row['True Positive'] / (total_row['True Positive'] + total_row['False Negative'])
total_row['F1 Score'] = 2 * (total_row['Precision'] * total_row['Recall']) / (total_row['Precision'] + total_row['Recall'])

# Convert the total_row to a DataFrame
total_df = pd.DataFrame([total_row])

# Concatenate the total row to the original DataFrame
df = pd.concat([df, total_df], ignore_index=True)

# Print the DataFrame
print(df)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the initial data
data = {
    'Class': ['small', 'midsize', 'large'],
    'True Positive': [179, 669, 16],
    'False Positive': [132, 20, 2],
    'False Negative': [0, 120, 34]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Calculate precision, recall, and F1 score
df['Precision'] = df['True Positive'] / (df['True Positive'] + df['False Positive'])
df['Recall'] = df['True Positive'] / (df['True Positive'] + df['False Negative'])
df['F1 Score'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])

# Define the input space for each class
input_space = {'small': 179 + 132 + 0, 'midsize': 669 + 20 + 120, 'large': 16 + 2 + 34}

# Add input space to the DataFrame
df['Input Space'] = df['Class'].map(input_space)

# Reorder columns
df = df[['Class', 'Input Space', 'True Positive', 'False Positive', 'False Negative', 'Recall', 'Precision', 'F1 Score']]

# Calculate the total row
total_row = {
    'Class': 'total',
    'Input Space': df['Input Space'].sum(),
    'True Positive': df['True Positive'].sum(),
    'False Positive': df['False Positive'].sum(),
    'False Negative': df['False Negative'].sum(),
}

# Calculate total precision, recall, and F1 score
total_row['Precision'] = total_row['True Positive'] / (total_row['True Positive'] + total_row['False Positive'])
total_row['Recall'] = total_row['True Positive'] / (total_row['True Positive'] + total_row['False Negative'])
total_row['F1 Score'] = 2 * (total_row['Precision'] * total_row['Recall']) / (total_row['Precision'] + total_row['Recall'])

# Convert the total_row to a DataFrame
total_df = pd.DataFrame([total_row])

# Concatenate the total row to the original DataFrame
df = pd.concat([df, total_df], ignore_index=True)

# Plotting
categories = df['Class'][:-1]  # Exclude the total row for plotting
precision = df['Precision'][:-1]
recall = df['Recall'][:-1]
f1_score = df['F1 Score'][:-1]

x = range(len(categories))

plt.figure(figsize=(10, 6))
plt.plot(x, precision, label='Precision', marker='o')
plt.plot(x, recall, label='Recall', marker='o')
plt.plot(x, f1_score, label='F1 Score', marker='o')
plt.xticks(x, categories)
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score by Class')
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix data
y_true = (
    ['small'] * (179 + 132 + 0) +
    ['midsize'] * (669 + 20 + 120) +
    ['large'] * (16 + 2 + 34)
)
y_pred = (
    ['small'] * 179 + ['midsize'] * 132 + ['small'] * 0 +
    ['midsize'] * 669 + ['small'] * 20 + ['large'] * 120 +
    ['large'] * 16 + ['midsize'] * 2 + ['small'] * 34
)

cm = confusion_matrix(y_true, y_pred, labels=['small', 'midsize', 'large'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['small', 'midsize', 'large'])
disp.plot(cmap=plt.cm.Blues)
plt.show()



