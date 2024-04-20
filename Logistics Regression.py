#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Downloads/50_Startups.csv')


# In[3]:


df


# In[4]:


df.isna().sum()


# In[10]:


df.columns


# In[13]:


X = df.drop(labels=['STATE'],axis=1)


# In[14]:


X


# In[15]:


Y = df[['STATE']]


# In[18]:


Y


# In[19]:


from sklearn.preprocessing import LabelEncoder,StandardScaler


# In[20]:


le = LabelEncoder()


# In[23]:


ss = StandardScaler()


# In[27]:


Y['STATE']=pd.DataFrame(le.fit_transform(Y['STATE']))


# In[28]:


Y


# In[29]:


X=pd.DataFrame(ss.fit_transform(X),columns=X.columns)
X


# In[31]:


from sklearn.model_selection import train_test_split
xtrain , xtest ,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=21)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[32]:


model= lr.fit(xtrain,ytrain)


# In[33]:


tr_pred = model.predict(xtrain)


# In[34]:


ts_pred = model.predict(xtest)


# In[39]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[40]:


tr_score = accuracy_score(ytrain,tr_pred)


# In[41]:


ts_score = accuracy_score(ytest,ts_pred)


# In[42]:


print(tr_score)


# In[43]:


print(ts_score)


# In[45]:


met = confusion_matrix(ytrain,tr_pred)


# In[46]:


met


# In[ ]:





