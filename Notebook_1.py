#!/usr/bin/env python
# coding: utf-8

# # Credit Card Lead Prediction

# In[2]:


#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# In[3]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[4]:


import xgboost as xgb


# In[5]:


#Loading the train CSV file into data frame
lead = pd.read_csv('train_s3TEQDk.csv')
lead.head()


# In[6]:


#Checking shape of the train file
lead.shape


# In[7]:


#Missing values findings
lead.isnull().sum().sort_values(ascending = False)


# In[8]:


# Checking for Nan values
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : lead[colname].isnull().sum() for colname in lead.columns}
Counter(x).most_common()


# In[9]:


#Dropping NAN values
data = lead.dropna() 


# In[10]:


#Missing values findings
data.isnull().sum().sort_values(ascending = False)


# In[11]:


# Checking for Nan values
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : data[colname].isnull().sum() for colname in data.columns}
Counter(x).most_common()


# In[12]:


#Categorical features for lead dataset
cat_features = data.select_dtypes(include='object').columns.tolist()
cat_features


# In[13]:


#Plotting the target variable
import seaborn as sns
sns.countplot(x=data.Is_Active, data=data)
plt.show()


# In[14]:


# Subset categorical features from lead data
data_cat = data[cat_features] 
data_cat.head()


# In[15]:



#Label encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(data)
df.head()


# In[16]:



#data fitting into label encoding
data = df.apply(LabelEncoder().fit_transform)
data


# In[17]:


# Extract numerical features and combine into the encoded categorical data
num_features = ['Age', 'Vintage', 'Avg_Account_Balance', 'Is_Lead']
data_num = data[num_features] 
data_proc = pd.concat([data_num, data], axis=1)
print(data_proc.shape)
print(data_num.shape)
print(data.shape)


# In[18]:


# Group sum and mean bureau data by ID
data_proc_sum = data_proc.groupby(['ID']).sum()
data_proc_mean = data_proc.groupby(['ID']).mean()


# In[19]:


data_proc_final = pd.merge(data_proc_sum, data_proc_mean, how='left', on='ID')


# In[20]:


data_proc_final.to_csv('data_proc_final.csv', index=True)


# In[21]:


data_proc_final = pd.read_csv('data_proc_final.csv')
data_proc_final.head()


# ## Process application with train/test

# In[22]:


credit_train = pd.read_csv('train_s3TEQDk.csv')
credit_test = pd.read_csv('test_mSzZ8RL.csv')


# In[23]:


# Check if 'ID' in the train data
'ID' in credit_train.columns


# In[24]:


# Merge data_proc_final to credit data
train_merged = pd.concat([credit_train, data_proc_final],axis=1)
test_merged = pd.concat([credit_test, data_proc_final],axis=1)
print(credit_train.shape)
print(train_merged.shape)
print(credit_test.shape)
print(test_merged.shape)


# In[25]:


# Drop ID columns for training
train_merged.drop('ID', axis=1, inplace=True)
test_merged.drop('ID', axis=1, inplace=True)


# In[26]:


# Extract numerical and categorical features for further processing: impute, scaling, one-hot encoding
num_features_1 = train_merged.select_dtypes(include='int64').columns.tolist()
num_features_1.remove('Is_Lead')


# In[27]:


num_features_2 = train_merged.select_dtypes(include='float64').columns.tolist()
cat_features = train_merged.select_dtypes(include='object').columns.tolist()


# In[28]:


num_features = num_features_1 + num_features_2
features = num_features + cat_features


num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
       # ('scaler', MinMaxScaler())  
        ('scaler', StandardScaler())  
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)


# In[29]:


preprocessor.fit(train_merged[features])
X_train = preprocessor.transform(train_merged[features])
X_test = preprocessor.transform(test_merged[features])

y_train = train_merged.Is_Lead.values

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape: ', X_test.shape)


# ## Train XGB model

# In[29]:


xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc", max_depth=4,learning_rate=0.277, gamma=0.382)
xgb_model.fit(X_train, y_train)


# In[30]:


# Calculate training accuracy
print(roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:,1]))


# In[31]:


predicted_customers =xgb_model.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[32]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_xgb.csv', index=False)


# ##  Random forrest

# In[33]:


# a simple RandomForrest Classifier without CV
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
#roc_auc_score(y_train, rf.predict_proba(X_train)[:,1])


# In[34]:


# Calculate training accuracy
print(roc_auc_score(y_train, rf.predict_proba(X_train)[:,1]))


# In[35]:


predicted_customers =rf.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[36]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_rf.csv', index=False)


# ## Random forrest with a cross validation

# In[30]:


rf_cv = RandomForestClassifier()
scores = cross_val_score(rf_cv, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
scores


# In[31]:


rf_cv.fit(X_train, y_train)
roc_auc_score(y_train, rf_cv.predict_proba(X_train)[:,1])


# In[32]:


predicted_customers =rf_cv.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[33]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_rfcv.csv', index=False)


# ## Logistic Regression

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression(C=10, tol=0.01, solver='lbfgs', max_iter=10000)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)


# In[35]:


# Calculate training accuracy
print(roc_auc_score(y_train, lr.predict_proba(X_train)[:,1]))


# In[36]:


predicted_customers =lr.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[37]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_lr.csv', index=False)


# ## Desiscion Tree

# In[38]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)


# In[39]:


# Calculate training accuracy
print(roc_auc_score(y_train, classifier.predict_proba(X_train)[:,1]))


# In[40]:


predicted_customers =classifier.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[41]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_dt.csv', index=False)


# ## Classification and regression tree

# In[42]:


from sklearn import tree # for decision tree models  
  # Fit the model
cart = tree.DecisionTreeClassifier( )
cart.fit(X_train, y_train)


# In[43]:


# Calculate training accuracy
print(roc_auc_score(y_train, cart.predict_proba(X_train)[:,1]))


# In[44]:


predicted_customers =cart.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[45]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_cart.csv', index=False)


# ## Naive Bayes

# In[46]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nb.predict(X_test)


# In[47]:


# Calculate training accuracy
print(roc_auc_score(y_train, nb.predict_proba(X_train)[:,1]))


# In[48]:


predicted_customers =nb.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[49]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_nb.csv', index=False)


# ## Gradient Boost

# In[50]:


# import machine learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# train with Gradient Boosting algorithm
# compute the accuracy scores on train and validation sets when training with different learning rates

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train, y_train)
    
predictions = gb.predict(X_test)


# In[51]:


# Calculate training accuracy
print(roc_auc_score(y_train, gb.predict_proba(X_train)[:,1]))


# In[52]:


predicted_customers =gb.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[53]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_gb.csv', index=False)


# ## Light gradient boost using cross validation

# In[54]:


get_ipython().system('pip install lightgbm')


# In[55]:


from lightgbm import LGBMClassifier
import lightgbm as lgb
lgb_cv = LGBMClassifier()
scores_lgb = cross_val_score(lgb_cv, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
scores_lgb


# In[56]:


lgb_cv.fit(X_train, y_train)
roc_auc_score(y_train, lgb_cv.predict_proba(X_train)[:,1])


# In[57]:


predicted_customers =lgb_cv.predict(X_test [0:105312])
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_customers)


# In[58]:


my_submission = pd.DataFrame({'ID': credit_test.ID, 'Is_Lead': predicted_customers})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_lgbcv.csv', index=False)


# In[ ]:




