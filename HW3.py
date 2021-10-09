#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from pandas.core.common import SettingWithCopyWarning
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

f = open('新竹_2019.csv')
hsinchu = pd.read_csv(f)
data=hsinchu.iloc[4825:6481,:]
data2=data.iloc[:,3:27]
for i in range(24):
    data2.loc[data2[str(i)].str.contains('#|x|A|\*'),str(i)]="NaN"
for i in range(24):
    data2[str(i)]=data2[str(i)].astype('float')
data2=data2.interpolate(axis=1)
#a=data2.loc[data2['13'].str.contains('NaN')]
#print(a)
data2.head(10)
month_data10 = []
sub_data = np.empty([18, 31*24])
for day in range(31):
    sub_data[:, day*24:(day+1)*24] = data2.iloc[(0*18*31+day*18):(0*18*31+(day+1)*18), :]
month_data10.append(sub_data)
month_data11 = []
sub_data = np.empty([18, 30*24])
for day in range(30):
    sub_data[:, day*24:(day+1)*24] = data2.iloc[(1*18*30+day*18):(1*18*30+(day+1)*18), :]
month_data11.append(sub_data)
mon10=pd.DataFrame(np.concatenate(month_data10))
mon11=pd.DataFrame(np.concatenate(month_data11))
x=pd.concat([mon10, mon11], axis=1)
month_data12 = []
sub_data = np.empty([18, 31*24])
for day in range(30):
    sub_data[:, day*24:(day+1)*24] = data2.iloc[(2*18*31+day*18):(2*18*31+(day+1)*18), :]
month_data12.append(sub_data)
y=pd.DataFrame(np.concatenate(month_data12))
x=x.interpolate(axis=1)
y=y.interpolate(axis=1)
#pm2.5/未來1小時預測(randomforestregression,linearregression)
x_train , y_train,x_test,y_test = [], [], [], []
for i in range(2):
    sub_data = x.iloc[9,:]
    for j in range(31*24-6):
        x_train.append(sub_data.iloc[j:j+6])
        y_train.append(sub_data.iloc[j+6])
sub_data = y.iloc[9,:]
for j in range(31*24-6):
    x_test.append(sub_data.iloc[j:j+6])
    y_test.append(sub_data.iloc[j+6])
y_train=np.array(y_train)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
rf = RandomForestRegressor(n_estimators = 50,random_state=50)
rf.fit(x_train, y_train);
test_y_predicted = rf.predict(x_test)
maerf = mean_absolute_error(y_test, test_y_predicted)
print("randomforest regression mae: "+str(maerf))
lreg = LinearRegression().fit(x_train, y_train)
test_y_predicted = lreg.predict(x_test)
maelreg = mean_absolute_error(y_test, test_y_predicted)
print("linear regression mae: "+str(maelreg))


# In[10]:


#pm2.5/未來6小時預測
x_train , y_train,x_test,y_test = [], [], [], []
for i in range(2):
    sub_data = x.iloc[9,:]
    for j in range(31*24-11):
        x_train.append(sub_data.iloc[j:j+6])
        y_train.append(sub_data.iloc[j+11])
sub_data = y.iloc[9,:]
for j in range(31*24-11):
    x_test.append(sub_data.iloc[j:j+6])
    y_test.append(sub_data.iloc[j+11])
y_train=np.array(y_train)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
rf = RandomForestRegressor(n_estimators = 50,random_state=40)
rf.fit(x_train, y_train);
test_y_predicted = rf.predict(x_test)
maerf = mean_absolute_error(y_test, test_y_predicted)
print("randomforest regression mae: "+str(maerf))
lreg = LinearRegression().fit(x_train, y_train)
test_y_predicted = lreg.predict(x_test)
maelreg = mean_absolute_error(y_test, test_y_predicted)
print("linear regression mae: "+str(maelreg))


# In[11]:


#18種屬性/未來1小時預測
x_train,y_train,x_test,y_test=[],[],[],[]
for i in range(2):
    sub_data = x.iloc[9,:]
    for j in range(31*24-6):
        x_train.append(x.iloc[:,j:j+6])
        y_train.append(sub_data.iloc[j+6])
sub_data = y.iloc[9,:]
for j in range(31*24-6):
    x_test.append(y.iloc[:,j:j+6])
    y_test.append(sub_data.iloc[j+6])
    
y_train=np.array(y_train)

y_test=np.array(y_test)

for i in range(1476):
    x_train[i]=np.array(x_train[i])
    x_train[i]=x_train[i].reshape(1,108)

xtrain=pd.DataFrame(x_train[0])
for i in range(1,1476):
    xtrain=np.concatenate([xtrain,x_train[i]],axis=0)
xtrain=pd.DataFrame(xtrain)

for i in range(738):
    x_test[i]=np.array(x_test[i])
    x_test[i]=x_test[i].reshape(1,108)
xtest=pd.DataFrame(x_test[0])
for i in range(1,738):
    xtest=np.concatenate([xtest,x_test[i]],axis=0)
xtest=pd.DataFrame(xtest)                              

rf = RandomForestRegressor(n_estimators = 50,random_state=50)
rf.fit(xtrain, y_train);
test_y_predicted = rf.predict(xtest)
maerf = mean_absolute_error(y_test, test_y_predicted)
print("randomforest regression mae: "+str(maerf))
lreg = LinearRegression().fit(xtrain, y_train)
test_y_predicted = lreg.predict(xtest)
maelreg = mean_absolute_error(y_test, test_y_predicted)
print("linear regression mae: "+str(maelreg))


# In[12]:


#18種屬性/未來6小時預測

x_train,y_train,x_test,y_test=[],[],[],[]
for i in range(2):
    sub_data = x.iloc[9,:]
    for j in range(31*24-11):
        x_train.append(x.iloc[:,j:j+6])
        y_train.append(sub_data.iloc[j+11])
sub_data = y.iloc[9,:]
for j in range(31*24-11):
    x_test.append(y.iloc[:,j:j+6])
    y_test.append(sub_data.iloc[j+11])
    
y_train=np.array(y_train)

y_test=np.array(y_test)

for i in range(1466):
    x_train[i]=np.array(x_train[i])
    x_train[i]=x_train[i].reshape(1,108)

xtrain=pd.DataFrame(x_train[0])
for i in range(1,1466):
    xtrain=np.concatenate([xtrain,x_train[i]],axis=0)
xtrain=pd.DataFrame(xtrain)

for i in range(733):
    x_test[i]=np.array(x_test[i])
    x_test[i]=x_test[i].reshape(1,108)
xtest=pd.DataFrame(x_test[0])
for i in range(1,733):
    xtest=np.concatenate([xtest,x_test[i]],axis=0)
xtest=pd.DataFrame(xtest)                              

rf = RandomForestRegressor(n_estimators = 50,random_state=50)
rf.fit(xtrain, y_train);
test_y_predicted = rf.predict(xtest)
maerf = mean_absolute_error(y_test, test_y_predicted)
print("randomforest regression mae: "+str(maerf))
lreg = LinearRegression()
lreg.fit(xtrain, y_train)
test_y_predicted = lreg.predict(xtest)
maelreg = mean_absolute_error(y_test, test_y_predicted)
print("linear regression mae: "+str(maelreg))


# In[ ]:




