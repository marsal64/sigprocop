#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
dataset = pd.read_csv('DataRaw.csv')
dataset = dataset.loc[(dataset.mm >= 55) & (dataset.mm <= 145)]
dataset.reset_index(inplace=True, drop=True)

dataset = dataset.dropna()
dataset


# In[36]:


dataset.shape


# In[50]:


data = dataset.sample(frac=0.1, random_state=786)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
data.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[51]:


from pycaret.regression import *


# In[99]:


exp_reg101 = setup(data = data, 
                   transformation=False, 
                   target = 'mm', 
                   polynomial_features=False,
                   session_id=123, 
                   feature_ratio = False,
                   feature_selection=False,
                   ignore_features=['offsetx']
                  )


# In[100]:


exp_reg101


# In[101]:


compare_models()


# In[102]:


#create_model.__doc__
#help(tune_model)
regr=tune_model('et', n_iter=100)


# In[103]:


regr_bagged = ensemble_model(regr)


# In[109]:


et = create_model('et')
#catboost = create_model('catboost')
#blend_all = blend_models([met, mcatboost])


# In[110]:


#stacknet = create_stacknet([[et, mcatboost], [et]])


# In[111]:


plot_model(et, 'feature')


# In[112]:


interpret_model(et)


# In[113]:


interpret_model(et, plot = 'correlation')


# In[114]:


interpret_model(et, plot = 'reason', observation = 10)


# In[121]:


pred=predict_model(et, data=data_unseen)
pred


# In[120]:


get_ipython().run_line_magic('pinfo', 'predict_model')


# In[124]:


save_model(et, 'savedmodelet')
et_saved = load_model('savedmodelet')


# In[125]:


save_experiment('experiment')


# In[126]:


experiment_loaded = load_experiment('experiment')

