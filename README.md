# Customer-Churn-Analysis
# 📊 Telco Customer Churn Analysis

This project focuses on analyzing and predicting customer churn for a telecom company using Python and machine learning.

## 🔍 Objective
To explore customer behavior and predict whether a customer will churn or stay based on various features such as contract type, internet service, monthly charges, and more.

## 🧰 Tools & Libraries Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📁 Dataset
- `Telco-Customer-Churn.csv`  
- Contains information on customer services, account information, and churn status.

## ✅ Steps Followed
1. **Data Loading & Cleaning**  
   - Converted `TotalCharges` to numeric  
   - Dropped missing or irrelevant values like `customerID`

2. **Exploratory Data Analysis (EDA)**  
   - Visualized churn distribution  
   - Analyzed churn by contract type and internet service

3. **Feature Encoding**  
   - Converted categorical variables to numerical using label encoding / one-hot encoding

4. **Model Building**  
   - Trained Logistic Regression and Decision Tree models  
   - Evaluated accuracy and model performance

## 📈 Sample Output

- **Churn Distribution** shows imbalance (more "No" than "Yes")
- **Contract Type**: Month-to-month users churn more
- **Internet Service**: Fiber optic users are more likely to churn
- **Model Accuracy**: ~78% using Decision Tree

## 🧠 What I Learned
- Real-world data cleaning and preprocessing
- Exploratory data analysis with plots
- Building and evaluating binary classification models
- Practical experience as a Data Analyst

## 📌 Author
**Asna Sidhique**

---

