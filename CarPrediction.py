#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("D:\\Data Science\\python practice\\Linear Regression\\car data.csv")


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


print(data['Fuel_Type'].unique())
print(data['Seller_Type'].unique())
print(data['Transmission'].unique())


# In[6]:


data.describe()


# In[7]:


data.columns


# In[8]:


final_data=data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[9]:


final_data.head()


# In[10]:


final_data['current_year']=2020


# In[11]:


final_data.head()


# In[12]:


final_data['no_years']=final_data['current_year']-final_data['Year']


# In[13]:


final_data.head()


# In[14]:


final_data.drop(['Year'],axis=1,inplace=True)


# In[15]:


final_data.head()


# In[16]:


final_data.drop(['current_year'],axis=1,inplace=True)


# In[17]:


final_data.head()


# In[18]:


final_data=pd.get_dummies(final_data,drop_first=True)


# In[19]:


final_data.head()


# In[20]:


final_data.corr()


# In[21]:


import seaborn as sns


# In[22]:


sns.pairplot(final_data)


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


corrmat=final_data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_data[top_corr_features].corr(),annot=True)


# In[25]:


X=final_data.drop(['Selling_Price'],axis=1)
Y=final_data['Selling_Price']


# In[26]:


X.head()


# In[27]:


Y.head()


# In[28]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,Y)


# In[29]:


print(model.feature_importances_)


# In[30]:


feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


# In[33]:


X_train.shape


# # Random Forest Regressor

# In[34]:


from sklearn.ensemble import RandomForestRegressor


# In[35]:


rf_random=RandomForestRegressor()


# In[36]:


import numpy as np


# In[37]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[38]:


from sklearn.model_selection import RandomizedSearchCV


# In[39]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[40]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[41]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[42]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[43]:


rf_random.fit(X_train,y_train)


# In[44]:


rf_random.best_params_


# In[45]:


rf_random.best_score_


# In[46]:


predictions=rf_random.predict(X_test)


# In[47]:


sns.distplot(y_test-predictions)


# In[48]:


plt.scatter(y_test,predictions)


# In[49]:


from sklearn import metrics


# In[50]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[53]:


from sklearn.metrics import r2_score


# In[54]:


print("Accuracy",r2_score(y_test,predictions))


# In[55]:


from sklearn.linear_model import LinearRegression


# In[56]:


mod=LinearRegression()
mod.fit(X_train,y_train)


# In[58]:


y_pred=mod.predict(X_test)


# In[59]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[60]:


print("Accuracy",r2_score(y_test,y_pred))


# In[62]:


from sklearn.ensemble import GradientBoostingRegressor


# In[64]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[65]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[66]:


gb_mod=GradientBoostingRegressor()


# In[67]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = gb_mod, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[68]:


rf_random.fit(X_train,y_train)


# In[69]:


rf_random.best_params_


# In[70]:


predictions=rf_random.predict(X_test)


# In[71]:


sns.distplot(y_test-predictions)


# In[72]:


plt.scatter(y_test,predictions)


# In[73]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Accuracy",r2_score(y_test,y_pred))


# In[ ]:




