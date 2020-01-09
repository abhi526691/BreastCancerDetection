#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df = pd.read_csv('Music/breastcancer/data.csv')


# In[10]:


df.head()


# In[11]:


#Check the no of nan value
df.isna().sum()


# In[12]:


df.head()


# In[15]:


df = df.dropna(axis=1)


# In[17]:


df.shape


# In[18]:


df.dtypes


# In[19]:


df['diagnosis']


# In[20]:


df['diagnosis'].value_counts()


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


LabelEncoder_Y = LabelEncoder()


# In[27]:


df.iloc[:,1] = LabelEncoder_Y.fit_transform(df.iloc[:,1].values)


# In[28]:


df.head()


# In[29]:


sns.countplot(df['diagnosis'], label='count')


# In[32]:


#check the pairs 
sns.pairplot(df.iloc[:,1:6], hue='diagnosis')


# In[33]:


#get the correlation of the columns


# In[37]:


df.corr()


# In[40]:


df.iloc[:,1:12].corr()


# In[50]:


sns.heatmap(df.iloc[:,1:12].corr(),annot=True,fmt='.0%',)
plt.figure(figsize=(30,30))


# In[51]:


#Splitting the data


# In[52]:


#INDEPENDENT VARIABLE = X 
#DEPENDENT VARIABLE = Y


# In[58]:


X = df.iloc[:,2:].values


# In[59]:


X


# In[60]:


Y = df.iloc[:,1].values


# In[61]:


Y


# In[74]:


from sklearn.model_selection import train_test_split


# In[83]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


# In[84]:


from sklearn.preprocessing import StandardScaler


# In[85]:


sc = StandardScaler()


# In[86]:


X_train = sc.fit_transform(X_train)


# In[88]:


X_test = sc.fit_transform(X_test)


# In[89]:


from sklearn.linear_model import LogisticRegression


# In[90]:


log = LogisticRegression()


# In[91]:


log.fit(X_train, Y_train)


# In[92]:


log.score(X_train, Y_train)


# In[93]:


pred = log.predict(X_test)


# In[94]:


pred


# In[95]:


Y_test


# In[96]:


from sklearn.neighbors import KNeighborsClassifier


# In[97]:


knn = KNeighborsClassifier()


# In[98]:


knn.fit(X_train, Y_train)


# In[99]:


knn.score(X_train, Y_train)


# In[101]:


predknn = knn.predict(X_test)


# In[102]:


predknn


# In[103]:


Y_test


# In[104]:


from sklearn.metrics import classification_report,accuracy_score


# In[105]:


classification_report(Y_test,log.predict(X_test))


# In[106]:


accuracy_score(Y_test, log.predict(X_test))


# In[107]:


classification_report(Y_test, predknn)


# In[108]:


accuracy_score(Y_test,pred)


# In[ ]:




