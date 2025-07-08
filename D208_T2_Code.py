import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import churn_clean.csv specifically bringing in only the columns needed for this analysis
customerdata = pd.read_csv('churn_clean.csv',usecols=['Children','Age','Income','Gender','Contract','PaymentMethod','Tenure','MonthlyCharge', 'Churn'])  

# Create a new object that omits null rows
filtered_customerdata = customerdata.dropna()

# Create object that displays the summary statistics for each numerical variable
summary_stats = filtered_customerdata.describe()[['Children', 'Age', 'Income', 'Tenure','MonthlyCharge']]

# Create and display histograms for each numerical variable
sns.histplot(data=filtered_customerdata['Tenure'], binwidth=12, kde=False)
sns.histplot(data=filtered_customerdata['Age'], binwidth=10, kde=False)
sns.histplot(data=filtered_customerdata['Income'], binwidth=10, kde=False)
sns.histplot(data=filtered_customerdata['MonthlyCharge'], binwidth=10, kde=False)

# Create a new data frame in preparation for one hot encoding
customerdata_ohe = filtered_customerdata 

# Select categorical variables and store them in a list
categorical_cols = ['Churn','PaymentMethod','Contract','Gender']

# Create an object that passes in the categorical variables and the data frame with new columns that have been one hot encoded with integers
col_ohe = pd.get_dummies(filtered_customerdata[categorical_cols], drop_first=True, dtype=int) 

# Creates a new object that concatenates the one hot encoded columns and removes the original columns and Churn
customerdata_ohe = pd.concat((customerdata_ohe, col_ohe), axis=1).drop(['Churn','PaymentMethod','Contract','Gender'], axis=1) 

# Creates a dictionary with the column names and the new column names
#https://www.geeksforgeeks.org/how-to-create-a-dictionary-in-python/
dict = {'Contract_One year':'Contract_1y','Contract_Two Year': 'Contract_2y', 'PaymentMethod_Electronic Check': 'PaymentMethod_E_Check', 'PaymentMethod_Mailed Check': 'PaymentMethod_Mailed_Check', 'PaymentMethod_Credit Card (automatic)': 'PaymentMethod_Credit_Card_Auto'} 

# Renames the columns in the data frame
customerdata_ohe.rename(columns=dict, inplace=True) 

# Initial logistic regression model using the 'Churn_Yes' variable as the response variable and prints output
mdl_customerdata = logit('Churn_Yes ~ Age + Children + PaymentMethod_Credit_Card_Auto + PaymentMethod_E_Check + PaymentMethod_Mailed_Check + Contract_1y + Contract_2y + Gender_Male + Gender_Nonbinary + MonthlyCharge + Income', data=customerdata_ohe).fit()
print(mdl_customerdata.summary())

# Confusion matrix for the original logistic regression model 
#TP FP
#FN TN
conf_matrix = mdl_customerdata.pred_table()
print(conf_matrix)

# Prepares the data for the RFE model
X = customerdata_ohe.drop('Churn_Yes', axis=1).values
y = customerdata_ohe['Churn_Yes'].values

# Scaling X data so features are in similar scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Regularization type of L1 is implemented with an increase in iterations from 100 to 500 and the solve of 'liblinear' is called to handle the L1 penalty.
#Citation: (Filho, 2023)
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X_scaled, y)
model.fit(X_scaled, y)

# Selects the features identified by the RFE algorithm, identifies their column names
selected_features = rfe.support_
reduced_features = (customerdata_ohe.drop('Churn_Yes', axis=1)).columns[selected_features]

# Creates a new data frame with the reduced features and prints the results
reduced_mdl_customerdata = logit('Churn_Yes ~ Tenure + PaymentMethod_E_Check +  Contract_1y + Contract_2y + MonthlyCharge', data=customerdata_ohe).fit()
print(reduced_mdl_customerdata.summary())

# Evaluate the model and print accuracy as a percentage
#https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Confusion matrix for the reduced model
reduced_conf_matrix = reduced_mdl_customerdata.pred_table()
print(reduced_conf_matrix)
