import pandas as pd

# Since the CSV is in the same folder, just use the filename
df = pd.read_csv("Telco-Customer-Churn.csv")

# Display first few rows
print(df.head())
# Show basic structure of the dataset
print("\nData Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Quick stats summary
print("\nSummary Statistics:")
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

# Set theme
sns.set(style="whitegrid")

# Churn count plot
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df, palette='pastel')
plt.title('Churn Distribution')
plt.show()

# Churn by Contract Type
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Contract Type')
plt.show()

# Churn by Internet Service
plt.figure(figsize=(8,5))
sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set1')
plt.title('Churn by Internet Service')
plt.show()

# Churn by Monthly Charges
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Churn vs Monthly Charges')
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.show()
# Convert target column 'Churn' to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert 'SeniorCitizen' to category (0/1 is fine)
df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

# Encode categorical variables using get_dummies
df_encoded = pd.get_dummies(df, drop_first=True)

# Show new shape
print("\nEncoded data shape:", df_encoded.shape)
from sklearn.model_selection import train_test_split

# Features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape)
print("Test size:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Predict
tree_preds = tree_model.predict(X_test)

# Evaluate
print("\n--- Decision Tree Results ---")
print("Accuracy:", accuracy_score(y_test, tree_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, tree_preds))
print("\nClassification Report:\n", classification_report(y_test, tree_preds))
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- DECISION TREE MODEL ---
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

tree_preds = tree_model.predict(X_test)

print("\n--- Decision Tree Results ---")
print("Accuracy:", accuracy_score(y_test, tree_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, tree_preds))
print("\nClassification Report:\n", classification_report(y_test, tree_preds))
