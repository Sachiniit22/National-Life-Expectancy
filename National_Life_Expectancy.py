#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# In[2]:


df = pd.read_csv(r'D:\work\who_life_exp.csv')


# # knowing the data

# In[3]:


df.shape


# In[4]:


df.head(50)


# In[5]:


df.info()


# In[6]:


df.describe()


# # data cleaning

# In[7]:


# drop the column which have higher null values
df.drop(['hospitals','une_poverty','une_literacy','une_school'],axis = 1, inplace = True)


# In[8]:


df.columns


# In[9]:


# drop effectless column
df.drop(['country_code'],axis = 1, inplace = True)


# In[10]:


# drop those rows which dont have atleast 18 non-missing values
df = df.dropna(thresh = 18)
df.reset_index(inplace = True)


# In[11]:


df.shape


# # correlations

# In[12]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), cmap = 'YlGnBu', annot = True)
plt.show()


# In[13]:


# we see that columns "une_gni" , "une_life" ,"life_exp60" , "age1-4mort" ,"polio", "diphtheria" and "une_infant" can be disregarded as they have coreelation >0.85 with their corresponding features.
df.drop(['index','une_gni','une_life','une_infant','life_exp60','age1-4mort','polio','diphtheria'],axis = 1, inplace = True)


# In[14]:


df.shape


# In[15]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), cmap = 'YlGnBu', annot = True)
plt.show()


# In[16]:


pd.DataFrame((df.corr()['life_expect'].drop('life_expect')*100).sort_values(ascending=False)).plot.bar(figsize = (12,8))


# In[17]:


n=1
plt.figure(figsize=(18,20))
for column in df.describe().columns:
    plt.subplot(7,3,n)
    n=n+1
    sns.boxplot(df[column]) 
    plt.tight_layout()


# In[18]:


df.columns


# In[19]:


a = df.groupby('region')
a.mean()


# In[20]:


a.median()


# In[21]:


for column in df[['alcohol', 'bmi', 'doctors',  ]]:
    df[column] = df.groupby(['region'])[column].transform(lambda region: region.fillna(region.mean()))


# In[22]:


for column in df[['adult_mortality','infant_mort','age5-19thinness','age5-19obesity','hepatitis', 'measles', 'basic_water','gghe-d',
       'che_gdp', 'une_pop','une_edu_spend']]:
    df[column] = df.groupby(['region'])[column].transform(lambda region: region.fillna(region.median()))


# In[23]:


df.info()


# In[24]:


df.isnull().sum()


# # fill null values by regression model

# In[25]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


df['gni_capita'].isnull().sum()


# In[28]:


# fill missing value of 'gni_capita' by linear regression model
data = df.copy()
from sklearn.model_selection import train_test_split
data.drop(['une_hiv','country','region'], axis=1,inplace=True)
data.dropna(inplace=True)
x = data.drop('gni_capita',axis=1)
sc.fit(x)
x = sc.transform(x)
y = data['gni_capita']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import PolynomialFeatures
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
model = LinearRegression().fit(x_, y_train)
y_pred = model.predict(x_)
print(r2_score(y_train,y_pred))
x_test = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_test)
y_test_pred = model.predict(x_test)
print(r2_score(y_test,y_test_pred))


# In[29]:


data2 = df[df['gni_capita'].isnull()]
data2.drop(['gni_capita','une_hiv','country','region'],axis=1,inplace=True)
x_null = PolynomialFeatures(degree=2, include_bias=False).fit_transform(data2)
df['gni_capita'][df['gni_capita'].isnull()] = model.predict(x_null)


# In[30]:


df['gni_capita'].isnull().sum()


# In[31]:


# fill missing value of 'une_hiv' by linear regression model
data = df.copy()
data.dropna(inplace=True)
x = data.drop(['une_hiv','country','region'],axis=1)
sc.fit(x)
x = sc.transform(x)
y = data['une_hiv']
lr = LinearRegression()
lr.fit(x,y)
ypred = lr.predict(x)
r2_score(y,ypred)


# In[32]:


data2 = df[df['une_hiv'].isnull()]
data2.drop(['une_hiv','country','region'],axis=1,inplace=True)
df['une_hiv'][df['une_hiv'].isnull()] = lr.predict(data2)


# In[33]:


df.info()


# In[34]:


df.isnull().sum()


# In[35]:


a = df.groupby(['region','year'])['life_expect'].mean()
a = pd.DataFrame(a)
# a.head(50)
# # df.set_index('year',inplace = True)
# # df.groupby('region')['life_expect'].plot(legend = True)


# # Distribution of class label

# In[36]:


plt.figure(figsize=(12,8))
sns.distplot(df['life_expect'], color="#008dce")


# # one hot encoding

# In[37]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False,drop = 'first')
finaldf = pd.DataFrame(enc.fit_transform(df[['region','country']]))


# In[38]:


final_df = pd.merge(df.drop(['region','country'],axis = 1),finaldf,left_index = True,right_index = True)
final_df.sample(5)


# # MODEL BUILDING

# In[39]:


x = final_df.drop('life_expect',axis=1)


# In[40]:


sc.fit(x)
x = sc.transform(x)


# In[41]:


y = final_df['life_expect']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
lr = LinearRegression()
model = lr.fit(X_train,y_train)
ypred = lr.predict(X_train)
print(r2_score(y_train,ypred))


# In[43]:


ypred = lr.predict(X_test)
print(r2_score(y_test,ypred))


# In[44]:


model.coef_


# In[45]:


print(f"intercept: {model.intercept_}")


# In[46]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,ypred)


# In[47]:


plt.scatter(y_test, ypred)
plt.xlabel("y_test")
plt.ylabel("Ypred")
plt.show()


# 
# # ANN model

# In[48]:


import tensorflow as tf


# In[49]:


# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
 
# Quick sanity check with the shapes of Training and testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[50]:


from keras.models import Sequential
from keras.layers import Dense
 
# create ANN model
model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=5, input_dim=204, kernel_initializer='normal', activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
 
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal',activation = 'linear'))
 
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mape'])
 
# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 10, epochs = 100, verbose=1)


# In[51]:


# Generating Predictions on testing data
Predictions = model.predict(X_test)
Predictions


# In[52]:


print(r2_score(y_test,Predictions))


# In[53]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,Predictions)


# In[54]:


df.groupby('year')['life_expect'].mean().plot()
plt.xlabel('year')
plt.ylabel('life expectancy')


# In[55]:


x = pd.DataFrame(df.groupby(['region'])['life_expect'].mean())
plt.bar(x.index.values,x['life_expect'],color = 'g')
plt.xticks(rotation = 90)


# In[56]:


x = pd.DataFrame(df.groupby(['region','year'])['life_expect'].mean())
x.reset_index(inplace = True)


# In[57]:


sns.barplot(x='year',y = 'life_expect',hue='region',data=x)
plt.ylim(50,85)
sns.set(rc={'figure.figsize':(100,7)})


# In[ ]:




