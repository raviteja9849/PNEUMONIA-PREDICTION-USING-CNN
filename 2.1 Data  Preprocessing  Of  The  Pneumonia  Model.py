#!/usr/bin/env python
# coding: utf-8

# In[1]:


from  keras.preprocessing.image   import  ImageDataGenerator


# In[ ]:





# In[3]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:





# In[5]:


x_train = train_datagen.flow_from_directory(r'C:\Users\Ravi Teja\Documents\data  for  project\train set', target_size=(64,64), batch_size=32, class_mode='categorical')


# In[6]:


x_train.class_indices


# In[9]:


x_test = test_datagen.flow_from_directory(r'C:\Users\Ravi Teja\Documents\data  for  project\test set', target_size=(64,64), batch_size=32, class_mode='categorical')


# In[10]:


x_train.image_shape


# In[ ]:




