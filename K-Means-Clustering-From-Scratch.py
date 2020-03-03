#!/usr/bin/env python
# coding: utf-8

# ## MENDOZA- HW 5
# Francis Mendoza
# 
# ASUID: 1213055998
# 
# CSE494- Intro To Machine Learning, F, 15:05-17:45 hrs MST
# 
# Professor Maneparambil
# 
# Assignment 5

# ## PROBLEM 1: K MEANS CLUSTERING FROM SCRATCH
# 
# * x_i = {(2 8), (3 3), (1, 2), (5, 8), (7, 3), (6, 4), (8, 4), (4, 7)}
# * k = 3, with three centers varying

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
#, mpld3
from matplotlib import style
import pandas as pd 

style.use('ggplot')

import pandas as pd


# In[23]:


class K_Means:
    def __init__(self, paramTup, k =3, tolerance = 0.0001, max_iterations = 500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = paramTup[i]


    def fit(self, data):
        #(!!!) CHOSE FIRST K ELEMENTS AS CENTROIDS. MANUALLY INITIALIZE!!
        #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        

        #begin iterations
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            #find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            #average the cluster datapoints to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)

            isOptimal = True

            for centroid in self.centroids:

                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False

            #break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
            if isOptimal:
                break

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# In[31]:


def main():
    filepath = 'KMC.csv'
    df = pd.read_csv(filepath, sep=',')

    #Check output of CSV
    print(df.shape)
    df

    df = df[['X', 'Y']]
    dataset = df.astype(float).values.tolist()

    #Return Numpy Array From CSV
    X = df.values
    X
    
    
    tripleCent = [[(2,8), (3,3), (5,8)], [(2,8),(3,3),(6,4)], [(2,8),(1,2),(6,4)]]
    for i in range(3):
        km = K_Means(tripleCent[i], 3)
        km.fit(X)
        print("FINAL CENTROIDS ARE: " + str(km.centroids))

        # Plotting starts here, the colors
        colors = 10*["r", "g", "c", "b", "k"]

        for centroid in km.centroids:
            plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s = 130, marker = "x")

        for classification in km.classes:
            color = colors[classification]
            for features in km.classes[classification]:
                plt.scatter(features[0], features[1], color = color,s = 30)

        print("KMC graph for " + str(tripleCent[i]))
        plt.show()
    
if __name__ == "__main__":
    main()


# # PROBLEM 2: PRINCIPAL COMPONENT ANALYSIS
# * Calculate covariance matrix
# * Calculate eigen values and eigen vectors (can use math libraries)
# * Graph principal components from feature space

# In[25]:


import pandas as pd 
filepath2 = 'pca_data2.csv'
df2 = pd.read_csv(filepath2, sep=',')
#df2

column1 = df2.x
column2 = df2.y

#column1
#column2


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,8)

#Separate into columns
x = column1
y = column2
X = np.vstack((x,y)).T

plt.scatter(X[:, 0], X[:, 1])
plt.title('PCA Data')
plt.axis('equal');


# In[27]:


# Covariance
def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)

# Covariance matrix
def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1])],                      [cov(X[1], X[0]), cov(X[1], X[1])]])

# Calculate covariance matrix 
print("COVARIANCE MATRIX")
comat = cov_mat(X.T) 
comat


# In[42]:


#Eigenvalue and Eigenvector calculation
print("EIGENVALUES are first array, EIGENVECTORS are second array")
np.linalg.eig(comat)


# In[40]:


#Plotting PCA from graph
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import seaborn as sns; sns.set()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    color='b',
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    
#Plot Data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');

print("PCA COMPONENTS: ")
print(pca.components_)

print("PCA EXPLAINED VARIANCES: ")
print(pca.explained_variance_)


# In[ ]:




