#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv("C:\\Users\\TANMOY DE\\Downloads\\churn-bigml-80.csv")

# Glance at the first five records
dataset.head()

# Print all the features of the data
dataset.columns


# In[2]:


# Churners vs Non-Churners
dataset['Churn'].value_counts()


# In[3]:


# Group data by 'Churn' and compute the mean
print(dataset.groupby('Churn')['Customer service calls'].mean())


# In[4]:


# Count the number of churners and non-churners by State
print(dataset.groupby('State')['Churn'].value_counts())


# In[5]:


# Import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of 'Total day minutes'
plt.hist(dataset['Total day minutes'], bins = 100)

# Display the plot
plt.show()


# In[6]:


# Create the box plot
sns.boxplot(x = 'Churn',
			y = 'Customer service calls',
			data = dataset,
			sym = "",				
			hue = "International plan")
# Display the plot
plt.show()


# In[7]:


# Features and Labels
X = dataset.iloc[:, 0:19].values
y = dataset.iloc[:, 19].values # Churn

# Encoding categorical data in X
from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])

labelencoder_X_2 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])

# Encoding categorical data in y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[8]:


# Removing extra column to avoid dummy variable trap
X_State = pd.get_dummies(X[:, 0], drop_first = True)

# Converting X to a dataframe
X = pd.DataFrame(X)

# Dropping the 'State' column
X = X.drop([0], axis = 1)

# Merging two dataframes
frames = [X_State, X]
result = pd.concat(frames, axis = 1, ignore_index = True)

# Final dataset with all numeric features
X = result


# In[9]:


# Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
													test_size = 0.2,
													random_state = 0)


# In[10]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[11]:


# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Instantiate the classifier
clf = RandomForestClassifier()

# Fit to the training data
clf.fit(X_train, y_train)


# In[12]:


# Predict the labels for the test set
y_pred = clf.predict(X_test)


# In[13]:


# Compute accuracy
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[14]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:




