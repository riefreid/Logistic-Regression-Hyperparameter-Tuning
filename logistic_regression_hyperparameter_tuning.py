import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('dataset.csv')

# Store the class labels
class_labels = data['label']  # Assuming the column with class labels is named 'label'

# Drop 'Barcode' and 'label' columns and the first row (column headers)
data = data.drop(columns=['Barcode', 'label'])
data = data.iloc[1:]  # Assuming the first row contains headers, adjust if necessary

# Convert the data to numeric (if needed)
data = data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, handle non-numeric values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Ensure shapes are aligned
class_labels = class_labels.iloc[1:]  # Adjusting class labels to match the remaining data after cleaning

# Splitting the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, class_labels, test_size=0.2, random_state=0, stratify=class_labels)

# Hyperparameter combinations to be tested
multi_class_options = ['ovr', 'multinomial']
penalty_options = ['l1', 'l2', 'elasticnet']

results = {}

# Iterate through different hyperparameters
for multi_class in multi_class_options:
    for penalty in penalty_options:
        if penalty == 'elasticnet':
            lr = LogisticRegression(multi_class=multi_class, penalty=penalty, solver='saga', l1_ratio=0.5, max_iter=1500, tol=0.001)  # Adjust tol
        else:
            lr = LogisticRegression(multi_class=multi_class, penalty=penalty, solver='saga', max_iter=1500, tol=0.001)  # Adjust tol

        # Fit the model
        lr.fit(x_train, y_train)

        # Predict using the test set
        predictions = lr.predict(x_test)

        # Calculate accuracy
        accuracy = lr.score(x_test, y_test)

        # Store the results
        results[(multi_class, penalty)] = accuracy

# Plotting the results
fig, ax = plt.subplots()
bars = ax.bar(range(len(results)), results.values(), align='center')

ax.set_xticks(range(len(results)))
ax.set_xticklabels(list(results.keys()), rotation=45)
ax.set_xlabel('Hyperparameters')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Different Hyperparameters')

# Display the numerical value on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Hyperparameter combinations to be tested
l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Range of L1 ratios

results = {}

# Iterate through different L1 ratios for elastic penalty with "OVR" multi-class option
for l1_ratio in l1_ratios:
    lr = LogisticRegression(multi_class='ovr', penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, max_iter=1500, tol=0.001)
    
    # Fit the model
    lr.fit(x_train, y_train)

    # Predict using the test set
    predictions = lr.predict(x_test)

    # Calculate accuracy
    accuracy = lr.score(x_test, y_test)

    # Store the results
    results[l1_ratio] = accuracy

# Find the best performing L1 ratio
best_l1_ratio = max(results, key=results.get)
best_accuracy = results[best_l1_ratio]

print(f"The best performing L1 ratio with 'OVR' multi-class option: {best_l1_ratio}")
print(f"Corresponding accuracy: {best_accuracy}")

# Hyperparameter combinations to be tested
l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Range of L1 ratios

results = {}

# Iterate through different L1 ratios for elastic penalty with "multinomial" multi-class option
for l1_ratio in l1_ratios:
    lr = LogisticRegression(multi_class='multinomial', penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, max_iter=1500, tol=0.001)
    
    # Fit the model
    lr.fit(x_train, y_train)

    # Predict using the test set
    predictions = lr.predict(x_test)

    # Calculate accuracy
    accuracy = lr.score(x_test, y_test)

    # Store the results
    results[l1_ratio] = accuracy

# Find the best performing L1 ratio
best_l1_ratio = max(results, key=results.get)
best_accuracy = results[best_l1_ratio]

print(f"The best performing L1 ratio: {best_l1_ratio}")
print(f"Corresponding accuracy: {best_accuracy}")
