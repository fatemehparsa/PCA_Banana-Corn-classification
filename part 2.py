#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
from PIL import Image
import glob
import scipy.misc
from resizeimage import resizeimage
import pandas as pd
from scipy import linalg as sl
#loading data sets and convert to vectors
Banana_im_list = []
banana_vec=[]
corn_im_list = []
corn_vec=[]
Kumquats_im_list = []
Kumquats_vec=[]
for filename in glob.glob('Banana Lady Finger/*.jpg'): #banana
    im=Image.open(filename).convert('RGBA')
    #im=im.resize((50,50))
    Banana_im_list.append(im)
    im_arr = np.array(im)
    banana_shape = im_arr.shape
    dim_1_arr = im_arr.ravel()
    vector = np.matrix(dim_1_arr)
    banana_vec.append(vector)
for filename in glob.glob('corn/*.jpg'): #corn
    im=Image.open(filename).convert('RGBA')
    #im=im.resize((50,50))
    corn_im_list.append(im)
    im_arr = np.array(im)
    corn_shape = im_arr.shape
    dim_1_arr = im_arr.ravel()
    vector = np.matrix(dim_1_arr)
    corn_vec.append(vector)
for filename in glob.glob('Kumquats/*.jpg'): #Kumquats
    im=Image.open(filename).convert('RGBA')
    #im=im.resize((50,50))
    Kumquats_im_list.append(im)
    im_arr = np.array(im)
    Kumquats_shape = im_arr.shape
    dim_1_arr = im_arr.ravel()
    vector = np.matrix(dim_1_arr)
    Kumquats_vec.append(vector)


# In[98]:


#convert vectors to dataframe matrix
banana_df = pd.DataFrame(banana_vec[0] ).T
for i in range(1,450):
    banana_df = pd.concat([banana_df,pd.DataFrame(banana_vec[i]).T], axis=1 )
banana_df = banana_df.T.reset_index(drop=True).T


corn_df = pd.DataFrame(corn_vec[0] ).T
for i in range(1,450):
    corn_df = pd.concat([corn_df,pd.DataFrame(corn_vec[i]).T], axis=1 )
corn_df = corn_df.T.reset_index(drop=True).T


Kumquats_df = pd.DataFrame(Kumquats_vec[0] ).T
for i in range(1,490):
    Kumquats_df = pd.concat([Kumquats_df,pd.DataFrame(Kumquats_vec[i]).T], axis=1 )
Kumquats_df = Kumquats_df.T.reset_index(drop=True).T


# In[ ]:





# In[106]:



def PCA(data, new_dim):
    m_data = data.mean(axis=0)
    data = data - m_data
    cov = np.cov(data, rowvar=False)
    e_val , e_vec = sl.eigh(cov)
    idx = np.argsort(e_val)[::-1]
    e_vec = e_vec[:,idx]
    e_val = e_val[idx]
    e_vec = e_vec[ : , : new_dim]
    return np.dot(e_vec.T, data.T).T
new_banana =PCA(banana_df,1000)
new_Kumquats =PCA(Kumquats_df,1000)
new_corn=PCA(corn_df,1000)


# In[107]:


for i in range (1):
    arr2 = np.asarray(new_banana[:,i]).reshape(banana_shape)
    img2 = Image.fromarray(arr2, 'RGBA')
    img2.show()
for i in range (1):
    arr2 = np.asarray(new_corn[:,i]).reshape(corn_shape)
    img2 = Image.fromarray(arr2, 'RGBA')
    img2.show()
for i in range (1):
    arr2 = np.asarray(new_Kumquats[:,i]).reshape(Kumquats_shape)
    img2 = Image.fromarray(arr2, 'RGBA')
    img2.show()


# In[ ]:





# In[ ]:




