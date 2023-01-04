# Jonathan Gaudaire (260966733)
# INSY 466 Individual project

# Import libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------- 1: CLASSIFICATION MODEL ---------------------------------

# Import data
df = pd.read_excel("D:\Documents\Travail\Mcgill\\5- Fall 2022\INSY 446 - Data mining for business analytics\Datasets\Kickstarter.xlsx")

# Pre-Processing

# Keep only rows with state successful or failed
df = df.loc[(df['state'] == 'successful') | (df['state'] == 'failed')]
# Drop columns that can't be know at the time of the submission
df = df.drop(columns=['pledged',
                      'state_changed_at',
                      'staff_pick',
                      'backers_count',
                      'usd_pledged',
                      'spotlight',
                      'state_changed_at_weekday',
                      'state_changed_at_month',
                      'state_changed_at_day',
                      'state_changed_at_yr',
                      'state_changed_at_hr',
                      'launch_to_state_change_days',
                      'launched_at',
                      'launched_at_yr',
                      'launched_at_month',
                      'launched_at_hr',
                      'launched_at_day',
                      'launched_at_weekday',
                      'create_to_launch_days',
                      'launch_to_deadline_days'],axis=1)
# Drop disable_communication (all values were == 'False')
df = df.drop('disable_communication',axis=1)
# Drop duplicates and missing values
df = df.drop_duplicates(subset=['name','goal'])
df = df.dropna()
# Get dummy variables for country and category
dummy_country = pd.get_dummies(df['country'])
dummy_category = pd.get_dummies(df['category'])
df = df.join(dummy_country)
df = df.drop('country',axis=1)
df = df.join(dummy_category)
df = df.drop('category',axis=1)
# Convert the goal variable to USD
df['goal_usd'] = df['goal']*df['static_usd_rate']
#df = df.drop(columns=['static_usd_rate'],axis=1)
# Reset index
df = df.reset_index(drop=True)
# Calculate the number of days between date created and deadline
from datetime import date
df['diff_created_deadline'] = 1
for i in range(len(df)):
     d0 = date(df['created_at_yr'][i], df['created_at_month'][i], df['created_at_day'][i])
     d1 = date(df['deadline_yr'][i], df['deadline_month'][i], df['deadline_day'][i])
     delta = d1-d0
     df['diff_created_deadline'][i] = delta.days

# Set up the variables

#Set up X (drop dependent variable)
X = df.drop(columns=['state',
                     'project_id',
                     'name',
                     'deadline',
                     'created_at',
                     'deadline_weekday',
                     'created_at_weekday',
                     # 'deadline_month',
                     'deadline_day',
                     # 'deadline_yr',
                     'created_at_day',
                     # 'created_at_month',
                     # 'created_at_yr',
                     'currency',
                     'static_usd_rate',
                     'created_at_hr',
                     'name_len_clean'])
# Set y = 0 for 'failed' and y = 1 for 'successful'
y = df["state"]
y = pd.get_dummies(y, drop_first=True).rename(columns={'successful':'state'})

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 8)

# Build the model (Decision Tree)
for i in range(1,16):
    decisiontree = DecisionTreeClassifier(max_depth=i)
    model1 = decisiontree.fit(X_train, y_train)
    # Using the model to predict the results based on the test dataset
    y_test_pred = model1.predict(X_test)
    # Calculate the mean squared error of the prediction
    print('Accuracy score Decision Tree(max depth='+str(i)+'): '+str(accuracy_score(y_test, y_test_pred)))

# Build the model (Logistic Regression)
y_train = y_train.squeeze() # Convert y_train to a 1D array
lr = LogisticRegression()
model2 = lr.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model2.predict(X_test)
# Calculate the mean squared error of the prediction
print(accuracy_score(y_test, y_test_pred))

#Build the model (KNN)
#Standardize the data
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)
X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 8)
y_train = y_train.squeeze()
for i in range (1,16):
    knn = KNeighborsClassifier(n_neighbors=i)
    model3 = knn.fit(X_train,y_train)
    y_test_pred = model3.predict(X_test)
    # Calculate the mean squared error of the prediction
    print('Accuracy score KNN(max depth='+str(i)+'): '+str(accuracy_score(y_test, y_test_pred)))

# Build the model (Random Forest)
y_train = y_train.squeeze()
for i in range (1,16):
    randomforest = RandomForestClassifier(random_state=5,max_features=i,n_estimators=300)
    model4 = randomforest.fit(X_train, y_train)
    y_test_pred = model4.predict(X_test)
    # Calculate the mean squared error of the prediction
    print('Accuracy score random forest(max depth='+str(i)+'): '+str(accuracy_score(y_test, y_test_pred)))
    
# Build the model (ANN)
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)
X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 8)
y_train = y_train.squeeze()
for i in range(1,16):
    mlp = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000, random_state=5)
    model5 = mlp.fit(X_train_std,y_train)
    y_test_pred = model5.predict(X_test_std)
    print('Accuracy score ANN(max depth='+str(i)+'): '+str(accuracy_score(y_test, y_test_pred)))
    
    
# Build the model with the highest accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 8)
y_train = y_train.squeeze()
randomforest = RandomForestClassifier(random_state=5,max_features=6,n_estimators=1000)
model_final = randomforest.fit(X_train, y_train)
y_test_pred_final = model_final.predict(X_test)
print(accuracy_score(y_test, y_test_pred_final))

# --------------------- 2 CLUSTERING ALGORITHM --------------------------------
# It is not specified in the guidelines I cannot use variables that cannot be known 
# at the time of the submission in the clustering model

# Set up the variables
df = pd.read_excel("D:\Documents\Travail\Mcgill\\5- Fall 2022\INSY 446 - Data mining for business analytics\Datasets\Kickstarter.xlsx")

# Pre-Processing
# Keep only rows with state successful or failed
df = df.loc[(df['state'] == 'successful') | (df['state'] == 'failed')]
df['goal_usd'] = df['goal']*df['static_usd_rate']
df = df.drop(columns=['state_changed_at',
                      'spotlight',
                      'state_changed_at_weekday',
                      'state_changed_at_month',
                      'state_changed_at_day',
                      'state_changed_at_yr',
                      'state_changed_at_hr',
                      'launch_to_state_change_days',
                      'launched_at',
                      'launched_at_yr',
                      'launched_at_month',
                      'launched_at_hr',
                      'launched_at_day',
                      'launched_at_weekday',
                      'create_to_launch_days',
                      'launch_to_deadline_days',
                      'disable_communication',
                      'project_id',
                      'deadline',
                      'created_at',
                      'deadline_weekday',
                      'created_at_weekday',
                      'deadline_month',
                      'deadline_day',
                      'deadline_yr',
                      'created_at_day',
                      'created_at_month',
                      'created_at_yr',
                      'currency',
                      'static_usd_rate',
                      'created_at_hr',
                      'name_len_clean',
                      'name_len',
                      'blurb_len',
                      'blurb_len_clean',
                      'deadline_hr',
                      'pledged',
                      'country',
                      'category',
                      'backers_count'],axis=1)

# Drop duplicates and missing values
df = df.drop_duplicates(subset=['name','goal'])
df =df.drop(columns=['name','goal'],axis=1)
df = df.dropna()
# Create dummy variable for the state variable
y = df["state"]
df = df.drop('state',axis=1)
y = pd.get_dummies(y, drop_first=True).rename(columns={'successful':'state'})
X = df.join(y)

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Build the model
for i in range(2,11):
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_std)
    labels = model.predict(X_std)
    print(silhouette_score(X_std,labels))

kmeans = KMeans(n_clusters=7)
model8 = kmeans.fit(X_std)
labels = model8.predict(X_std)

centroids = model8.cluster_centers_
X['ClusterMembership'] = labels
cluster_0 = X[X['ClusterMembership'] == 0].mean()
cluster_1 = X[X['ClusterMembership'] == 1].mean()
cluster_2 = X[X['ClusterMembership'] == 2].mean()
cluster_3 = X[X['ClusterMembership'] == 3].mean()
cluster_4 = X[X['ClusterMembership'] == 4].mean()
cluster_5 = X[X['ClusterMembership'] == 5].mean()
# cluster_6 = X[X['ClusterMembership'] == 6].mean()
# cluster_7 = X[X['ClusterMembership'] == 7].mean()
clusters = pd.concat([cluster_0,cluster_1,cluster_2,cluster_3,cluster_4,cluster_5],axis=1)




# ----------------------- GRADING ---------------------------------------------

# Import Grading Data
df_grading = pd.read_excel("D:\Documents\Travail\Mcgill\\5- Fall 2022\INSY 446 - Data mining for business analytics\Datasets\\Kickstarter-Grading-Sample.xlsx")

# Keep only rows with state successful or failed
df_grading = df_grading.loc[(df_grading['state'] == 'successful') | (df_grading['state'] == 'failed')]
# Drop columns that can't be know at the time of the submission
df_grading = df_grading.drop(columns=['pledged',
                      'state_changed_at',
                      'staff_pick',
                      'backers_count',
                      'usd_pledged',
                      'spotlight',
                      'state_changed_at_weekday',
                      'state_changed_at_month',
                      'state_changed_at_day',
                      'state_changed_at_yr',
                      'state_changed_at_hr',
                      'launch_to_state_change_days',
                      'launched_at',
                      'launched_at_yr',
                      'launched_at_month',
                      'launched_at_hr',
                      'launched_at_day',
                      'launched_at_weekday',
                      'create_to_launch_days',
                      'launch_to_deadline_days'],axis=1)
# Drop disable_communication (all values were == 'False')
df_grading = df_grading.drop('disable_communication',axis=1)
# Drop duplicates and missing values
df_grading = df_grading.drop_duplicates(subset=['name','goal'])
df_grading = df_grading.dropna()
# Get dummy variables for country and category
dummy_country = pd.get_dummies(df_grading['country'])
dummy_category = pd.get_dummies(df_grading['category'])
df_grading = df_grading.join(dummy_country)
df_grading = df_grading.drop('country',axis=1)
df_grading = df_grading.join(dummy_category)
df_grading = df_grading.drop('category',axis=1)
# Convert the goal variable to USD
df_grading['goal_usd'] = df_grading['goal']*df_grading['static_usd_rate']
#df = df.drop(columns=['static_usd_rate'],axis=1)
# Reset index
df_grading = df_grading.reset_index(drop=True)
# Calculate the number of days between date created and deadline
from datetime import date
df_grading['diff_created_deadline'] = 1
for i in range(len(df_grading)):
     d0 = date(df_grading['created_at_yr'][i], df_grading['created_at_month'][i], df_grading['created_at_day'][i])
     d1 = date(df_grading['deadline_yr'][i], df_grading['deadline_month'][i], df_grading['deadline_day'][i])
     delta = d1-d0
     df_grading['diff_created_deadline'][i] = delta.days

# Set up the variables

#Set up X (drop dependent variable)
X_grading = df_grading.drop(columns=['state',
                     'project_id',
                     'name',
                     'deadline',
                     'created_at',
                     'deadline_weekday',
                     'created_at_weekday',
                     # 'deadline_month',
                     'deadline_day',
                     # 'deadline_yr',
                     'created_at_day',
                     # 'created_at_month',
                     # 'created_at_yr',
                     'currency',
                     'static_usd_rate',
                     'created_at_hr',
                     'name_len_clean'])
# Set y = 0 for 'failed' and y = 1 for 'successful'
y_grading = df_grading["state"]
y_grading = pd.get_dummies(y_grading, drop_first=True).rename(columns={'successful':'state'})

# Apply the model previously trained to the grading data
y_grading_pred = model_final.predict(X_grading)

# Calculate the accuracy score
accuracy_score(y_grading, y_grading_pred)


