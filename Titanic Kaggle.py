#!/usr/bin/env python
# coding: utf-8

# In[339]:


import pandas as pd
df=pd.read_csv(r'/Users/sowmya/Downloads/titanic/train.csv',encoding='ISO-8859-1')


# In[340]:


df.head()


# In[341]:


df.describe()


# In[342]:


pd.value_counts(df['Survived'])


# In[343]:


pd.value_counts(df['Pclass'])


# In[344]:


df.dtypes


# In[345]:


df.head()


# In[346]:


df.head()


# In[347]:


import matplotlib.pyplot as plt
plt.hist(df['Age'],bins=6)
plt.show


# In[348]:


plt.hist(df['Fare'],bins=50)
plt.show


# In[349]:


pd.value_counts(df['Fare'])


# In[374]:


plt.hist(df['Fare'],bins=50)
plt.show


# In[375]:


import seaborn as sns
sns.boxplot(data=df,x=df['Fare'])


# In[351]:


df.drop(df.loc[df['Fare']>350].index, inplace=True)


# In[352]:


import seaborn as sns
sns.boxplot(data=df,x=df['Fare'])


# In[353]:


pd.value_counts(df['Parch'])


# In[354]:


pd.value_counts(df['SibSp'])


# In[355]:


df.isnull().sum()


# In[356]:


df1=df.drop(['Cabin'],axis=1)


# In[357]:


df1.head()


# In[358]:


df2=df1.dropna()


# In[359]:


df2.describe()


# In[360]:


df2.head(20)


# In[361]:


df2.tail(20)


# In[362]:


X=df3[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[363]:


Y=df3['Survived']


# In[364]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df3['Sex']=encoder.fit_transform(df3['Sex'])
df3['Embarked']=encoder.fit_transform(df3['Embarked'])


# In[365]:


df3.head()


# In[366]:


X=df3[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
Y=df3['Survived']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,train_size=0.7,random_state=42)
ytrain.shape


# In[367]:


from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1.fit(xtrain,ytrain)


# In[368]:


ypred=model1.predict(xtest)


# In[369]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(ypred,ytest)


# In[370]:


accuracy_score(ypred,ytest)


# In[371]:


from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier()
model2.fit(xtrain,ytrain)


# In[372]:


ypred2=model2.predict(xtest)


# In[373]:


accuracy_score(ypred2,ytest)


# In[299]:


dftest.head()


# In[300]:


dftest1=dftest.dropna()
Xtest=dftest1[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
dftest1.dropna()
dftest1.head()


# In[301]:


Xtest.head()


# In[302]:


Xtest.isnull().sum()


# In[303]:


Ypred=model1.predict(Xtest)


# In[204]:





# In[ ]:




