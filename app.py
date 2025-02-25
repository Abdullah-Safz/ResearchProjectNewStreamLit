#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



# In[3]:


# Data Pre procsssing


# In[4]:


# Assuming the dataset is stored as a CSV file.
# Replace with your actual dataset file path.

# Upload the file


# Assuming the file is named 'your_dataset.xlsx'
df = pd.read_excel('Data.xlsx')

# Display the first few rows of the dataset
print(df.head())



# In[5]:


df = df.drop(columns=['Investment Description', 'Investment Years'], axis = 1 )


# In[6]:


df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
print(df.columns)


# In[7]:


sns.countplot(data=df, x='Country', color='skyblue')

plt.title('Distribution of Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=90)  # Rotate x-axis labels if there are many countries
plt.show()


# In[8]:


print(df['Country'].unique)


# In[9]:


# Clean up the 'Country' column to standardize all instances of 'Sri Lanka'
df['Country'] = df['Country'].str.strip()  # Remove any leading/trailing spaces
df['Country'] = df['Country'].str.lower()  # Convert all text to lowercase

# Replace all variations of 'sri lanka' with the standard 'Sri Lanka'
df['Country'] = df['Country'].replace({
    'sri lanka': 'Sri Lanka',  # Standardize 'sri lanka' to 'Sri Lanka'
    'srilanka': 'Sri Lanka',
    'sri lanka ': 'Sri Lanka',  # In case there are extra spaces
    'sri lanka/': 'Sri Lanka',
    'sri lanka  ': 'Sri Lanka',
    'sri  lanka': 'Sri Lanka',
    'sri lanka?': 'Sri Lanka',
    '2584/08/10/2002': 'Sri Lanka',
    'united kindom' : 'united kingdom',
    'united state of america' : 'usa',
    'netherlands' : 'netherland',
    'united sates' : 'usa',
    'uk' : 'united kingdom',
    'dubai' : 'uae dubai',
    'uae uae dubai' : 'uae dubai',
    'uae uae dubai' : 'uae dubai',
}, regex=True)

# After replacing, you can check the cleaned data
print(df['Country'].value_counts())


# In[10]:


# Clean up the 'Country' column to standardize all instances of 'Sri Lanka'
#df['Satisfaction'] = df['Satisfaction'].str.strip()  # Remove any leading/trailing spaces
#df['Satisfaction'] = df['Satisfaction'].str.lower()  # Convert all text to lowercase

# Replace all variations of 'sri lanka' with the standard 'Sri Lanka'
df['Satisfaction'] = df['Satisfaction'].replace({
    1: 100,  # Standardize 'sri lanka' to 'Sri Lanka'

}, regex=True)

# After replacing, you can check the cleaned data
print(df['Satisfaction'].value_counts())


# In[11]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Country' column
df['Size of Investment_encoded'] = label_encoder.fit_transform(df['Size of Investment'])

# Display the first few rows of the dataframe to check the encoded column
print(df[['Size of Investment', 'Size of Investment_encoded']].head())

# Optional: Print the mapping of labels to country names
country_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label encoding mapping:", country_mapping)
df.head()


# In[12]:


from sklearn.impute import KNNImputer

# Initialize KNNImputer
imputer = KNNImputer(n_neighbors=5)

# Apply KNN imputation on the entire dataframe
df[['Satisfaction']] = imputer.fit_transform(df[['Satisfaction']])



# In[13]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns  # Select only numeric columns

# Initialize KNN Imputer
imputer = KNNImputer(n_neighbors=5)

# Apply imputer to numerical columns only
df[num_cols] = imputer.fit_transform(df[num_cols])

print(df)


# In[14]:


df[df.select_dtypes(include=['float']).columns] = df.select_dtypes(include=['float']).astype(int)

print(df)


# In[15]:


import pandas as pd
import numpy as np
from scipy import stats

# Function to check normality for each group using Shapiro-Wilk test
def check_normality(df, group_col, value_col):
    # Group the data by the group column
    grouped_data = df.groupby(group_col)[value_col].apply(list)
    
    normality_results = {}
    for group, values in grouped_data.items():
        # Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(values)
        normality_results[group] = {'Shapiro-Wilk Stat': stat, 'p-value': p_value}
        
        if p_value > 0.05:
            normality_results[group]['Normality'] = 'Normal'
        else:
            normality_results[group]['Normality'] = 'Not Normal'
    
    return normality_results

import pandas as pd
import scipy.stats as stats

def check_normality_with_encoding(df, categorical_col, numerical_col):
    # One-hot encoding for categorical variable
    df_encoded = pd.get_dummies(df, columns=[categorical_col], dtype=int)

    normality_results = {}

    for col in df_encoded.columns:
        if col.startswith(categorical_col + "_"):  # Only check one-hot encoded columns
            values = df[numerical_col][df_encoded[col] == 1]  # Select corresponding rows
            
            if len(values) > 3:  # Shapiro-Wilk needs at least 3 values
                stat, p_value = stats.shapiro(values)
                normality_results[col] = {'Shapiro-Wilk Stat': stat, 'p-value': p_value}
                
                normality_results[col]['Normality'] = 'Normal' if p_value > 0.05 else 'Not Normal'
    
    return normality_results



# Function to check if the distributions are the same across groups (using ANOVA or Kruskal-Wallis)
import pandas as pd
import numpy as np
from scipy import stats

def perform_anova(df, group_col, value_col):
    # Group the data by the group column and extract values
    grouped_data = df.groupby(group_col)[value_col].apply(list)
    values_list = [values for group, values in grouped_data.items()]

    # Perform ANOVA test (F-statistic and p-value)
    f_stat, p_value = stats.f_oneway(*values_list)

    # Return the results
    return {'ANOVA F-statistic': f_stat, 'p-value': p_value}


def perform_kruskal_wallis(df, group_col, value_col):
    # Group the data by the group column and extract values
    grouped_data = df.groupby(group_col)[value_col].apply(list)
    
    # Check if there are any groups with less than 2 data points
    for group, values in grouped_data.items():
        if len(values) < 2:
            return {'Error': f"Group '{group}' has less than 2 data points. Kruskal-Wallis test cannot be performed."}

    # Check for NaN or infinite values in the value column
    if df[value_col].isna().sum() > 0 or (df[value_col] == float('inf')).sum() > 0 or (df[value_col] == float('-inf')).sum() > 0:
        return {'Error': 'The value column contains NaN or infinite values. Please clean the data before performing the test.'}

    # Create the list of values for Kruskal-Wallis test
    values_list = [values for group, values in grouped_data.items()]

    # Perform Kruskal-Wallis test (H-statistic and p-value)
    h_stat, p_value = stats.kruskal(*values_list)

    # Determine if the p-value is statistically significant
    if p_value < 0.20:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"

    # Return the results
    return {
        'Kruskal-Wallis H-statistic': h_stat,
        'p-value': p_value,
        'Significance': significance
    }


import pandas as pd
import scipy.stats as stats

def kruskal_wallis_with_encoding(df, categorical_col, numerical_col):
    # One-hot encoding for categorical variable
    df_encoded = pd.get_dummies(df, columns=[categorical_col], dtype=int)

    kw_results = {}

    groups = []  # List to store numerical values for each category

    for col in df_encoded.columns:
        if col.startswith(categorical_col + "_"):  # Only check one-hot encoded columns
            values = df[numerical_col][df_encoded[col] == 1]  # Select corresponding rows
            
            if len(values) > 3:  # Kruskal-Wallis needs at least 3 values per group
                groups.append(values)
    
    if len(groups) > 1:  # Kruskal-Wallis requires at least 2 groups
        stat, p_value = stats.kruskal(*groups)
        kw_results['Kruskal-Wallis Stat'] = stat
        kw_results['p-value'] = p_value
        kw_results['Significant'] = 'Yes' if p_value < 0.20 else 'No'
    else:
        kw_results['Error'] = 'Not enough groups with sufficient data for Kruskal-Wallis test'

    return kw_results


# In[16]:


# Call the function with correct column names as strings

normality = check_normality_with_encoding( df,'Type of Ownership','Satisfaction')

# Display the results
print("Normality Test Results:")
print(normality)



# In[17]:


kruskal_result = kruskal_wallis_with_encoding(df, 'Type of Ownership', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[18]:


#Data must be at least length 3. there fore check normality by
normality = check_normality_with_encoding(df, 'Industry Sector', 'Satisfaction')

# Display the results
print("Normality Test Results:")
print(normality)


# In[19]:


kruskal_result = kruskal_wallis_with_encoding(df, 'Industry Sector', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[20]:


normality = check_normality(df, 'Size of Investment', 'Satisfaction')

# Display the results
print("Normality Test Results:")
print(normality)


# In[21]:


kruskal_result = perform_kruskal_wallis(df, 'Size of Investment_encoded', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[22]:


#Data must be at least length 3. there fore check normality by
normality = check_normality_with_encoding(df, 'Country', 'Satisfaction')

# Display the results
print("Normality Test Results:")
print(normality)


# In[23]:


kruskal_result = kruskal_wallis_with_encoding(df, 'Country', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[24]:


from scipy.stats import shapiro

stat, p = shapiro(df['Satisfaction'])
print(f"Shapiro-Wilk test p-value: {p}")

if p > 0.05:
    print("✅ Numerical variable is normally distributed.")
else:
    print("❌ Numerical variable is NOT normally distributed.")


# In[25]:


kruskal_result = perform_kruskal_wallis(df, 'Investor Facilitation Center', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[26]:


normality = check_normality(df, 'Environment', 'Satisfaction')

# Display the results
print("Normality Test Results:")
print(normality)


# In[27]:


kruskal_result = perform_kruskal_wallis(df, 'Environment', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[28]:


normality = check_normality(df, 'Engineering', 'Satisfaction')

# Display the results
print("Normality Test Results:")
print(normality)


# In[29]:


kruskal_result = perform_kruskal_wallis(df, 'Engineering', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[30]:


kruskal_result = perform_kruskal_wallis(df, 'Investment approvals', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[31]:


kruskal_result = perform_kruskal_wallis(df, 'External Line Agencies', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[32]:


kruskal_result = perform_kruskal_wallis(df, 'Legal Activities', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[33]:


kruskal_result = perform_kruskal_wallis(df, 'Implementation Period', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[34]:


kruskal_result = perform_kruskal_wallis(df, 'Issuing Tax Certificate', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[35]:


kruskal_result = perform_kruskal_wallis(df, 'Import & Export', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[36]:


kruskal_result = perform_kruskal_wallis(df, 'Industrial Relations', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[37]:


kruskal_result = perform_kruskal_wallis(df, 'Research and Policy', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[38]:


kruskal_result = perform_kruskal_wallis(df, 'IT', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[39]:


kruskal_result = perform_kruskal_wallis(df, 'Zonal Infrastructure', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[40]:


kruskal_result = perform_kruskal_wallis(df, 'Zonal Utility', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[41]:


kruskal_result = perform_kruskal_wallis(df, 'Zonal Waste Management', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[42]:


kruskal_result = perform_kruskal_wallis(df, 'Zonal Security & Fire Facilities', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[43]:


kruskal_result = perform_kruskal_wallis(df, 'Tax incentives to investors', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[44]:


kruskal_result = perform_kruskal_wallis(df, 'Government Policies', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[45]:


kruskal_result = perform_kruskal_wallis(df, 'Ease of doing business', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[46]:


kruskal_result = perform_kruskal_wallis(df, 'Skilled labor', 'Satisfaction')

# Output the result
print("Kruskal-Wallis Test Result:", kruskal_result)


# In[47]:


import numpy as np

# Check for NaN values
print(df.isna().sum())



# In[48]:


print(df['Satisfaction'].isna().sum())  # Check for NaNs
print(np.isinf(df['Satisfaction']).sum())  # Check for infinite values



# In[49]:


df = df.drop(columns=['Type of Ownership', 'Country', 'Industry Sector', 'Size of Investment', 'Size of Investment_encoded'], axis = 1 )


# In[50]:


df = df.drop('Zonal Security & Fire Facilities', axis = 1 )


# In[51]:


df = df.drop('Zonal Waste Management', axis = 1 )


# In[52]:


print(df.columns)


# In[53]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Define the features (X) and target (y)
X = df.drop('Satisfaction', axis=1)  # Drop the 'Satisfaction' column
y = df['Satisfaction']  # Target variable

# Initialize the imputer (e.g., fill NaN values with the mean of each column)
imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Convert the imputed X back to a DataFrame and keep the original column names
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Impute missing values for the target (y)
imputer = SimpleImputer(strategy='most_frequent')
y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()  # Use ravel() to flatten array

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the transformation function



# In[54]:


print("Number of features in X_train:", X_train.shape[1])



# In[55]:


# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict on the test set
y_pred = linear_model.predict(X_test)

# Evaluate the model
print("Linear Regression Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")


# In[56]:


# Initialize the Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Train the model
decision_tree_model.fit(X_train, y_train)

# Predict on the test set
y_pred_tree = decision_tree_model.predict(X_test)

# Evaluate the model
print("Decision Tree Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_tree)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_tree)}")
print(f"R-squared: {r2_score(y_test, y_pred_tree)}")


# In[57]:


# Initialize the Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
random_forest_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = random_forest_model.predict(X_test)
y_train_pred_rf = random_forest_model.predict(X_train)
# Evaluate the model
print("Random Forest Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"Test Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"Train Error: {mean_squared_error(y_train, y_train_pred_rf)}")
print(f"R-squared: {r2_score(y_test, y_pred_rf)}")


# In[58]:


import xgboost as xgb

xgboost_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Train the model
xgboost_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgboost_model.predict(X_test)
test_error = mean_squared_error(y_test,y_pred_xgb)

# Evaluate the model
print("XGBoost Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_xgb)}")
print(f"Test Error: {mean_squared_error(y_test, y_pred_xgb)}")
print(f"R-squared: {r2_score(y_test, y_pred_xgb)}")


# In[59]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the KNN regressor
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = knn_model.predict(X_test)
y_train_pred_knn = knn_model.predict(X_train)

# Evaluate the model
print("KNN Regressor Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_knn)}")
print(f"Test Error: {mean_squared_error(y_test, y_pred_knn)}")
print(f"Train Error: {mean_squared_error(y_train, y_train_pred_knn)}")
print(f"R-squared: {r2_score(y_test, y_pred_knn)}")


# In[60]:


from sklearn.svm import SVR

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

print("SVR Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_svr)}")
print(f"Test Error (MSE): {mean_squared_error(y_test, y_pred_svr)}")
print(f"R-squared: {r2_score(y_test, y_pred_svr)}")


# In[61]:


from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("Gradient Boosting Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_gb)}")
print(f"Test Error (MSE): {mean_squared_error(y_test, y_pred_gb)}")
print(f"R-squared: {r2_score(y_test, y_pred_gb)}")


# In[62]:


from sklearn.ensemble import AdaBoostRegressor

ab_model = AdaBoostRegressor(n_estimators=100, random_state=42)
ab_model.fit(X_train, y_train)
y_pred_ab = ab_model.predict(X_test)

print("AdaBoost Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_ab)}")
print(f"Test Error (MSE): {mean_squared_error(y_test, y_pred_ab)}")
print(f"R-squared: {r2_score(y_test, y_pred_ab)}")


# In[63]:


from sklearn.ensemble import ExtraTreesRegressor

et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)
y_pred_et = et_model.predict(X_test)

print("Extra Trees Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_et)}")
print(f"Test Error (MSE): {mean_squared_error(y_test, y_pred_et)}")
print(f"R-squared: {r2_score(y_test, y_pred_et)}")


# In[64]:


# Create a DataFrame to compare the models
def calculate_regression_accuracy(y_true, y_pred, threshold=0.1):
    """
    Custom accuracy for regression:
    Percentage of predictions that are within 'threshold' of the true values.
    E.g., threshold=0.1 means predictions within 10% of the true value are considered accurate.
    """
    accuracy = np.mean(np.abs((y_true - y_pred) / y_true) < threshold) * 100
    return accuracy

# Calculate accuracy for each model
accuracy_lr = calculate_regression_accuracy(y_test, y_pred, threshold=0.1)  # Linear Regression
accuracy_tree = calculate_regression_accuracy(y_test, y_pred_tree, threshold=0.1)  # Decision Tree
accuracy_rf = calculate_regression_accuracy(y_test, y_pred_rf, threshold=0.1)  # Random Forest
#accuracy_xg = calculate_regression_accuracy(y_test, y_pred_xgb, threshold=0.1)
# Create the model comparison DataFrame with custom accuracy
model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'R-squared': [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_tree),
        r2_score(y_test, y_pred_rf),
        #r2_score(y_test, y_pred_xgb)
    ],
    'Mean Squared Error': [
        mean_squared_error(y_test, y_pred),
        mean_squared_error(y_test, y_pred_tree),
        mean_squared_error(y_test, y_pred_rf),
        #mean_squared_error(y_test, y_pred_xgb)
    ],
    'Root Mean Squared Error (RMSE)': [
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred_tree)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        #np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    ],
    'Mean Absolute Error (MAE)': [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred_tree),
        mean_absolute_error(y_test, y_pred_rf),
        #mean_absolute_error(y_test, y_pred_xgb)
    ],
    'Accuracy (within 10%)': [
        accuracy_lr,
        accuracy_tree,
        accuracy_rf,
        #accuracy_xg

    ]
})

# Display the model comparison
print(model_comparison)


# In[65]:


print(df.columns)


# In[66]:


selected_columns = ['Investor Facilitation Center', 'Environment', 'Engineering',
       'Investment approvals', 'External Line Agencies', 'Legal Activities',
       'Implementation Period', 'Issuing Tax Certificate', 'Import & Export',
       'Industrial Relations', 'Research and Policy', 'IT',
       'Zonal Infrastructure', 'Zonal Utility', 'Tax incentives to investors',
       'Government Policies', 'Ease of doing business', 'Skilled labor']


# In[67]:


import optuna
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# Load your data here
# df = pd.read_csv('your_data.csv')

# Define the features (X) and target (y)
X = df.drop('Satisfaction', axis=1)  # Replace 'Satisfaction' with the name of your target column
y = df['Satisfaction']

# Step 1: Scale the Features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),  # Number of neighbors
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),  # Weight function
        'p': trial.suggest_int('p', 1, 2),  # Power parameter for the Minkowski distance metric
        'leaf_size': trial.suggest_int('leaf_size', 10, 50),  # Leaf size for KDTree or BallTree
    }

    # Initialize the KNN regressor with the selected parameters
    model = KNeighborsRegressor(**params)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test set
    preds = model.predict(X_test)
    
    # Return the R-squared score
    return r2_score(y_test, preds)

# Step 4: Create an Optuna study for maximizing the R-squared score
study = optuna.create_study(direction='maximize')

# Optimize the study for 20 trials
study.optimize(objective, n_trials=20)

# Output the best parameters found
print("Best Parameters:", study.best_params)

# Step 5: Retrain the model with the best hyperparameters found
best_knn = KNeighborsRegressor(**study.best_params)

# Fit the model on the training data
best_knn.fit(X_train, y_train)

# Step 6: Predict on the test set
y_pred = best_knn.predict(X_test)

# Step 7: Evaluate the model performance
print("Final Model Performance:")
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Step 8: Perform Cross-Validation for a more robust evaluation
cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validated R-squared scores: {cv_scores}")
print(f"Mean cross-validated R-squared: {cv_scores.mean()}")


# In[84]:


import numpy as np

# New data with 18 features (1 sample)
new_data = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

# Reshape the data to 2D: (1 sample, 18 features)
new_data = new_data.reshape(1, -1)

import joblib

# Load the saved model
loaded_knn_model = joblib.load('C:\\Users\\DELL\\Desktop\\Research project\\app\\ada_model.pkl')

# Now you can use `loaded_knn_model` just like your original model
# For example, to make predictions:
prediction = loaded_knn_model.predict(new_data)  # new_data should be properly preprocessed
print("Prediction:", prediction)



# In[76]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

# Initialize the AdaBoostRegressor
ada_model = AdaBoostRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'loss': ['linear', 'square']
}




# Perform RandomizedSearchCV
ada_random = RandomizedSearchCV(ada_model, param_grid, n_iter=20, cv=5, scoring='r2', random_state=42, n_jobs=-1)
ada_random.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", ada_random.best_params_)


# In[77]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Retrain the model with the best hyperparameters found
best_ada = AdaBoostRegressor(**ada_random.best_params_)  # Use the best parameters found from RandomizedSearchCV

# Fit the model on the training data
best_ada.fit(X_train, y_train)

# Predict on the test and train sets
y_pred_ada_test = best_ada.predict(X_test)
y_pred_ada_train = best_ada.predict(X_train)

# Evaluate the model performance on test data
print("Final Model Performance on Test Data:")
print(f"Test R-squared: {r2_score(y_test, y_pred_ada_test)}")
print(f"Test Mean Absolute Error: {mean_absolute_error(y_test, y_pred_ada_test)}")
print(f"Test Mean Squared Error: {mean_squared_error(y_test, y_pred_ada_test)}")

# Evaluate the model performance on train data
print("\nFinal Model Performance on Train Data:")
print(f"Train R-squared: {r2_score(y_train, y_pred_ada_train)}")
print(f"Train Mean Absolute Error: {mean_absolute_error(y_train, y_pred_ada_train)}")
print(f"Train Mean Squared Error: {mean_squared_error(y_train, y_pred_ada_train)}")


# In[78]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor

# Fit the AdaBoost model (assuming it's already trained)
best_ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.01, loss='linear', random_state=42)
best_ada.fit(X_train, y_train)

# Get feature importances
importances = best_ada.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Convert X_train to a DataFrame if it is a numpy ndarray
if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train, columns=selected_columns)

# Create a plot of feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (AdaBoost Regressor)")
plt.barh(range(X_train.shape[1]), importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()


# In[76]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the selected columns
selected_columns = [
    'Investor Facilitation Center', 'Environment', 'Engineering',
       'Investment approvals', 'External Line Agencies', 'Legal Activities',
       'Implementation Period', 'Issuing Tax Certificate', 'Import & Export',
       'Industrial Relations', 'Research and Policy', 'IT',
       'Zonal Infrastructure', 'Zonal Utility', 'Tax incentives to investors'
]

# Check if X_train is a numpy array, and if so, convert it to a DataFrame
if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train, columns=selected_columns)

# Check if all columns are numeric in X_train
for col in selected_columns:
    if col not in X_train.columns:
        print(f"Column '{col}' not found in X_train.")
    elif not pd.api.types.is_numeric_dtype(X_train[col]):
        print(f"Column '{col}' is not numeric.")

# Set up the figure size
plt.figure(figsize=(15, 12))

# Loop through each column and plot a histogram
for i, col in enumerate(selected_columns, 1):
    if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col]):  # Check if column exists and is numeric
        plt.subplot(4, 4, i)  # Arrange subplots in a 4x4 grid
        sns.histplot(data=X_train[col], bins=30, kde=True, color='blue')
        plt.title(col)
        plt.xlabel('')
    else:
        print(f"Skipping column '{col}'")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[79]:


import joblib

# Save the trained AdaBoost model
joblib.dump(best_ada, 'C:\\Users\\DELL\\Desktop\\Research project\\app\\ada_model.pkl')



print("Model saved successfully!")
#cd "C:\Users\DELL\Desktop\Research project\app"
#streamlit run app.py


# In[89]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('C:\\Users\\DELL\\Desktop\\Research project\\app\\ada_model.pkl')

# Set up the Streamlit page
st.title('ADA Regressor - Predict Satisfaction')
st.write('Enter the values for the features and get the predicted satisfaction score (from 0 to 100).')

# Define ordinal features
ordinal_features = [
    'Investor Facilitation Center', 'Environment', 'Engineering',
    'Investment approvals', 'External Line Agencies', 'Legal Activities',
    'Implementation Period', 'Issuing Tax Certificate', 'Import & Export',
    'Industrial Relations', 'Research and Policy', 'IT',
    'Zonal Infrastructure', 'Zonal Utility', 'Tax incentives to investors',
    'Government Policies', 'Ease of doing business', 'Skilled labor'
]

# Initialize session state only if it's not set
if "feature_values" not in st.session_state:
    st.session_state.feature_values = {feature: 3 for feature in ordinal_features}

# Function to reset all slider values
def refresh_values():
    st.session_state.feature_values = {feature: 3 for feature in ordinal_features}
    st.rerun()  #
# User input sliders
for feature in ordinal_features:
    st.session_state.feature_values[feature] = st.slider(
        feature, 3, 5, value=st.session_state.feature_values[feature]
    )

# Prediction button
if st.button('Predict'):
    # Convert session state values to a DataFrame
    input_data = pd.DataFrame([{feature: st.session_state.feature_values[feature] for feature in ordinal_features}])
    
    # Make prediction
    predicted_satisfaction = model.predict(input_data)
    
    # Display prediction
    st.write(f'Predicted Satisfaction Score: {predicted_satisfaction[0]:.2f}')

# Refresh button to reset all sliders
#if st.button("Refresh"):
    #refresh_values()  # 

