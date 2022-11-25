#!/usr/bin/env python
# coding: utf-8

# # Forecasting Assignment

# ### Data Set - CocaCola_Sales_Rawdata

# # 1. Import Necessary libraries

# In[1]:

import cv2

import os

import numpy as np

import pickle

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import models,utils

import pandas as pd

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import load_img,img_to_array

from tensorflow.python.keras import utils

current_path = os.getcwd()

# getting the current path

dog_breeds_category_path = os.path.join(current_path, 'static\dog_breeds_category.pickle')

# loading class_to_num_category

predictor_model = load_model(r'static\dogbreed.h5')

with open(dog_breeds_category_path, 'rb') as handle:

    dog_breeds = pickle.load(handle)

# loading the feature extractor model

feature_extractor = load_model(r'static\feature_extractor.h5')

def predictor(img_path): # here image is file name 

    img = load_img(img_path, target_size=(331,331))

    img = img_to_array(img)

    img = np.expand_dims(img,axis = 0)

    features = feature_extractor.predict(img)

    prediction = predictor_model.predict(features)*100

    prediction = pd.DataFrame(np.round(prediction,1),columns = dog_breeds).transpose()

    prediction.columns = ['values']

    prediction  = prediction.nlargest(5, 'values')

    prediction = prediction.reset_index()

    prediction.columns = ['name', 'values']

    return(prediction)

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # 2. Import Data

# In[2]:


sales_data = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')
sales_data


# # 3. Data Understanding

# ## 3.1 Initial Analysis :

# In[3]:


sales_data.head()


# In[4]:


sales_data.shape


# In[5]:


sales_data.info()


# In[6]:


sales_data.isna().sum()


# In[7]:


sales_data.describe()


# In[8]:


sales_data.dtypes


# In[9]:


sales_data.columns


# In[10]:


temp = sales_data.Quarter.str.replace(r'(Q\d)_(\d+)', r'19\2-\1')
sales_data['quater'] = pd.to_datetime(temp).dt.strftime('%b-%Y')
sales_data.head()


# In[11]:


sales_data = sales_data.drop(['Quarter'], axis = 1)
sales_data.reset_index(inplace=True)
sales_data['quater'] = pd.to_datetime(sales_data['quater'])
sales_data = sales_data.set_index('quater')
sales_data.head()


# ## 3.2 Visualization using Lineplot for Sales :

# In[12]:


sales_data['Sales'].plot(figsize = (15, 6))
plt.show()


# ## 3.3 Moving Average Method :

# In[13]:


for i in range(2,10,2):
    sales_data["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
plt.show()


# ## 3.4 Time series decomposition plot :

# In[14]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[15]:


ts_add = seasonal_decompose(sales_data.Sales,model = "additive")
fig = ts_add.plot()
plt.show()


# In[16]:


ts_mul = seasonal_decompose(sales_data.Sales,model = "multiplicative")
fig = ts_mul.plot()
plt.show()


# ## 3.5 Visualization using TSA Plot :

# In[17]:


import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models


# In[18]:


tsa_plots.plot_acf(sales_data.Sales)

tsa_plots.plot_pacf(sales_data.Sales)

plt.show()


# # 4.  Evaluation Metric RMSE

# In[19]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
warnings.filterwarnings('ignore')


# ## 4.1 Splitting Data :

# In[20]:


def RMSE(org, pred):
    rmse = np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse


# In[21]:


Train = sales_data.head(30)
Test = sales_data.tail(12)


# ## 4.2 Simple Exponential Method :

# In[22]:


simple_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_simple_model = simple_model.predict(start = Test.index[0],end = Test.index[-1])


# In[23]:


rmse_simple_model = RMSE(Test.Sales, pred_simple_model)
print('RMSE Value of Simple Exponential :',rmse_simple_model)


# ## 4.3 Holt method :

# In[24]:


holt_model = Holt(Train["Sales"]).fit()
pred_holt_model = holt_model.predict(start = Test.index[0],end = Test.index[-1])


# In[25]:


rmse_holt_model = RMSE(Test.Sales, pred_holt_model)
print('RMSE Value of Holt :',rmse_holt_model)


# ## 4.4 Holts winter exponential smoothing with additive seasonality and additive trend :

# In[26]:


holt_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal = "add",trend = "add",seasonal_periods = 4).fit()
pred_holt_add_add = holt_model_add_add.predict(start = Test.index[0],end = Test.index[-1])


# In[27]:


rmse_holt_add_add_model = RMSE(Test.Sales, pred_holt_add_add)
print('RMSE Value of Holts add and add :',rmse_holt_add_add_model)


# ## 4.5 Holts winter exponential smoothing with multiplicative seasonality and additive trend

# In[28]:


holt_model_multi_add = ExponentialSmoothing(Train["Sales"],seasonal = "mul",trend = "add",seasonal_periods = 4).fit() 
pred_holt_multi_add = holt_model_multi_add.predict(start = Test.index[0],end = Test.index[-1])


# In[29]:


rmse_holt_model_multi_add_model = RMSE(Test.Sales, pred_holt_multi_add)
print('RMSE Value of Holts Multi and add :',rmse_holt_model_multi_add_model)


# # 5. Model based Forecasting Methods

# ## 5.1 Data preprocessing for models :

# In[30]:


sales_data_1 = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')
sales_data_1.head()


# In[31]:


sales_data_2 = pd.get_dummies(sales_data_1, columns = ['Quarter'])
sales_data_2.columns = ['Sales','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1',
                        'Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2',
                        'Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3',
                        'Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4']
sales_data_2.head()


# In[32]:


t = np.arange(1,43)
sales_data_2['t'] = t
sales_data_2['t_squared'] = sales_data_2['t']*sales_data_2['t']
log_Sales = np.log(sales_data_2['Sales'])


# In[33]:


sales_data_2['log_Sales'] = log_Sales
sales_data_2.head()


# ## 5.2 Splitting Data :

# In[34]:


train, test = np.split(sales_data_2, [int(.67 *len(sales_data_2))])


# ## 5.3 Linear Model :

# In[35]:


import statsmodels.formula.api as smf 


# In[36]:


linear_model = smf.ols('Sales~t',data = train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))


# In[37]:


rmse_linear_model = RMSE(test['Sales'], pred_linear)
print('RMSE Value of Linear :',rmse_linear_model)


# ## 5.4 Exponential Model :

# In[38]:


Exp_model = smf.ols('log_Sales~t',data = train).fit()
pred_Exp = pd.Series(Exp_model.predict(pd.DataFrame(test['t'])))


# In[39]:


rmse_Exp_model = RMSE(test['Sales'], np.exp(pred_Exp))
print('RMSE Value of Exponential :',rmse_Exp_model)


# ## 5.5 Quadratic Model

# In[40]:


Quad_model= smf.ols('Sales~t+t_squared',data = train).fit()
pred_Quad = pd.Series(Quad_model.predict(test[["t","t_squared"]]))


# In[41]:


rmse_Quad_model = RMSE(test['Sales'], pred_Quad)
print('RMSE Value of Quadratic :',rmse_Quad_model)


# ## 5.6 Additive Model :

# In[42]:


additive_model =  smf.ols('Sales~ Q1+Q2+Q3+Q4',data = train).fit()
pred_additive = pd.Series(additive_model.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))


# In[43]:


rmse_additive_model = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_additive))**2))
print('RMSE Value of Additive :',rmse_additive_model)


# ## 5.7 Additive Linear Model :

# In[44]:


additive_linear_model = smf.ols('Sales~t+Q1+Q2+Q3+Q4',data = train).fit()
pred_additive_linear = pd.Series(additive_linear_model.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))


# In[45]:


rmse_additive_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_additive_linear))**2))
print('RMSE Value of Additive Linear :',rmse_additive_linear)


# ## 5.8 Additive Quadratic Model :

# In[46]:


additive_quad_model = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data = train).fit()
pred_additive_quad = pd.Series(additive_quad_model.predict(pd.DataFrame(test[['t','t_squared','Q1','Q2','Q3','Q4']])))


# In[47]:


rmse_additive_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_additive_quad))**2))
print('RMSE Value of Additive Quadratic :',rmse_additive_quad)


# ## 5.9 Multi Linear Model :

# In[48]:


multi_linear_model = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = train).fit()
pred_multi_linear = pd.Series(multi_linear_model.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))


# In[49]:


rmse_multi_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_multi_linear)))**2))
print('RMSE Value of Multi Linear :',rmse_multi_linear)


# ## 5.10 Multi Quadratic Model :

# In[50]:


multi_quad_model = smf.ols('log_Sales~t+t_squared+Q1+Q2+Q3+Q4',data = train).fit()
pred_multi_quad = pd.Series(multi_quad_model.predict(test[['t','t_squared','Q1','Q2','Q3','Q4']]))


# In[51]:


rmse_multi_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_multi_quad)))**2))
print('RMSE Value of Multi Quadratic :',rmse_multi_quad)


# # 6. ARIMA model

# In[52]:


series = sales_data_1.copy()
series


# ## 6.1 Separate out a validation dataset :

# In[53]:


split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header = False)
validation.to_csv('validation.csv', header = False)


# ## 6.2 Evaluate a Base model :

# In[54]:


X = sales_data_1['Sales'].values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]


# ## 6.3 Walk Farward Validation :

# In[55]:


from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[56]:


history = [x for x in train]
predictions = list()


# In[57]:


for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# In[58]:


rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE Value : %f' % rmse)


# In[59]:


rmse_Persistence_model = 565.7799


# # 7. Conclusion

# In[60]:


list = [['Simple Exponential Method',rmse_simple_model], ['Holt method',rmse_holt_model],
          ['Holt exp smoothing add',rmse_holt_add_add_model],['Holt exp smoothing multi',rmse_holt_model_multi_add_model],
          ['Linear Model',rmse_linear_model],['Exponential model',rmse_Exp_model],['Quadratic model',rmse_Quad_model],
          ['Additive Model',rmse_additive_model],['Additive Linear Model',rmse_additive_linear],
          ['Additive Qudratic Model',rmse_additive_quad],['Muli Linear Model',rmse_multi_linear],
          ['Multi Quadratic Model',rmse_multi_quad],
          ['Persistence/ Base model', rmse_Persistence_model]]


# In[61]:


df = pd.DataFrame(list, columns = ['Model', 'RMSE_Value']) 
df


# In[62]:


sns.barplot(data = df,x = 'Model',y = 'RMSE_Value')
plt.show()

