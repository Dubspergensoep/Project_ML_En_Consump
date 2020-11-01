#!/usr/bin/env python
# coding: utf-8

#  # Some Simple models

# Import modules

# In[86]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Read data

# In[2]:


df = pd.read_csv('consumption.csv')


# In[13]:


print(df)


# In[5]:


df.shape


# In[100]:


row=df.iloc[1]
begin,end=get_monthi(12)
print(row[begin:end])
print(row[begin:end].isnull().sum())
print(row[begin:end].values.any())
print(row[begin:end].mean())


# In[111]:


row=df.iloc[3247]
begin,end=get_monthi(11)
print(row[begin:end].shape)
print(end-begin)
print(row[begin:end].isnull().sum())


# In[112]:


# loop over all meter ids
for i in range(0,df.shape[0]):
    #load row
    meter=df.iloc[i]
    
    # loop over all months
    print("Meter i=%i has not NaN values in month:" %i)
    for m in range (1,13):
        ind_b,ind_e=get_monthi(m)               #get index of beginning and end of month
        month=meter[ind_b:ind_e]
        # Check if months has numeric values        
        n_NaN=month.isnull().sum() #Number of NaN's
        if n_NaN<(ind_e-ind_b) and m<12:
            print(m)


# In[3]:


def get_monthi(n):
    begin=48*31*(n-1)+1
    end=48*31*n
    if n==1:
        begin=1
    if n>1:
        end-=3*48
    if n>2:
        begin-=3*48
    if n>3:
        end-=48
    if n>4:
        begin-=48
    if n>5:
        end-=48
    if n>6:
        begin-=48
    if n>8:
        end-=48
    if n>9:
        begin-=48
    if n>10:
        end-=48
    if n>11:
        begin-=48
    return begin,end


# # Naive Drift

# Check performance naive drift

# In[4]:


pli=False #print log info
NaN_t=1200 #NaN threshold (entire month has about 1400 datapoint)->we want atleast 1/7th of a month in this case
totSSE=0
nSSE=0


# loop over all meter ids
for i in range(0,df.shape[0]):
    #load row
    meter=df.iloc[i]
    fmf=False #first month found -> this variable is needed because naive drift needs 2 data points
    
    # loop over all months
    for m in range (1,13):
        ind_b,ind_e=get_monthi(m)               #get index of beginning and end of month
        month=meter[ind_b:ind_e]

        # Check if months has numeric values        
        n_NaN=month.isnull().sum() #Number of NaN's
        if n_NaN<NaN_t:
            mean_month=month.mean()
            
            #determine naive drift
            if fmf:
                ndrift=2*mean_month-last_month
                #evaluate prediction
                if m <12:
                    next_ind_b,next_ind_e=get_monthi(m+1) 
                    next_month=meter[next_ind_b:next_ind_e]
                    mnm=next_month.mean()         #mean next month
                    SSE=(mnm-ndrift)**2
                    totSSE+=SSE
                    nSSE+=1
                    if pli:
                        print("i=%i m=%i SSE=%f" % (i, m, SSE))
            else:
                fmf=True
            
            last_month=mean_month
              
        
        if pli:     
            print("Current month is %i" % m)
            print("Amount of NaN found %i" %n_NaN)


# In[5]:


RMSE=np.sqrt(totSSE/nSSE)
print(RMSE)


# RMSE=0.06684300400401268
# Average half hourly consumption of a month is between 0.16 and 0.30. Then our relative error is about 20 to 37.5 % 

# # Linear Models

# First try a lineair model that is only dependent on temperature

# In[92]:


#load data
weather_avg = pd.read_csv('weather-avg.csv')
weather_min = pd.read_csv('weather-min.csv')
weather_max = pd.read_csv('weather-max.csv')


# In[93]:


print(weather_min)


# In[94]:


print(weather_avg.shape) #Containts average temperature of each day instead of half hour
print(weather_min.shape)
print(weather_max.shape)
print(df.shape)

#print(df.loc[:,"meter_id"])
#print(weather_avg.loc[:,"meter_id"])
meterid=df.loc[:,"meter_id"][5]
row_wa=weather_avg.loc[weather_avg['meter_id'] == meterid] #finds rown corresponding to the meter id 
print(row_wa)


# In[62]:


weather_avg.loc[:,"2017-01-01 00:00:00":"2017-12-31 00:00:00"].mean(1)
row_wa.loc[:,"2017-01-01 00:00:00":"2017-01-31 00:00:00"].mean(1)


# In[65]:


def get_mean_temp(row,month):
    if month==1:
        return row.loc[:,"2017-01-01 00:00:00":"2017-01-31 00:00:00"].mean(1)
    elif month==2:
        return row.loc[:,"2017-02-01 00:00:00":"2017-02-28 00:00:00"].mean(1)
    elif month==3:
        return row.loc[:,"2017-03-01 00:00:00":"2017-03-31 00:00:00"].mean(1)
    elif month==4:
        return row.loc[:,"2017-04-01 00:00:00":"2017-04-30 00:00:00"].mean(1)
    elif month==5:
        return row.loc[:,"2017-05-01 00:00:00":"2017-05-31 00:00:00"].mean(1)
    elif month==6:
        return row.loc[:,"2017-06-01 00:00:00":"2017-06-30 00:00:00"].mean(1)
    elif month==7:
        return row.loc[:,"2017-07-01 00:00:00":"2017-07-31 00:00:00"].mean(1)
    elif month==8:
        return row.loc[:,"2017-08-01 00:00:00":"2017-08-31 00:00:00"].mean(1)
    elif month==9:
        return row.loc[:,"2017-09-01 00:00:00":"2017-09-30 00:00:00"].mean(1)
    elif month==10:
        return row.loc[:,"2017-10-01 00:00:00":"2017-10-31 00:00:00"].mean(1)
    elif month==11:
        return row.loc[:,"2017-11-01 00:00:00":"2017-11-30 00:00:00"].mean(1)
    elif month==12:
        return row.loc[:,"2017-12-01 00:00:00":"2017-12-31 00:00:00"].mean(1)
    else:
        print("Error: this is not a valid input for month")
        


# In[74]:


print(get_mean_temp(row_wa,4))
meter=df.iloc[5]
ind_b,ind_e=get_monthi(12)               #get index of beginning and end of month
month=meter[ind_b:ind_e]
print(month.mean())


# In[20]:


for i in range(1,df.shape[0]):
    if df.loc[:,"meter_id"][i]!=weather_avg.loc[:,"meter_id"][i]:
        print("Meter id not equal at index %i" %i)
    


# In[ ]:





# It turns out that meter id are not the same for the rows between the different files :/ <br>
# solution is:  row_wa=weather_avg.loc[weather_avg['meter_id'] == meterid] #finds rown corresponding to the meter id <br>
# An improvement could be made by making it so and each meter exclusively belongs to the testing or training data, at the moment his is not the case

# In[95]:


temps=[]
temps_min=[]
temps_max=[]
En_con=[]

#for i in range(0,df.shape[0]):

for i in range(0,df.shape[0]):
    #load rows
    meter=df.iloc[i]
    meterid=df.loc[:,"meter_id"][5]
    row_wa=weather_avg.loc[weather_avg['meter_id'] == meterid] #finds rown corresponding to the meter id 
    row_wm=weather_min.loc[weather_min['meter_id'] == meterid]
    row_wM=weather_max.loc[weather_max['meter_id'] == meterid]
    
    # loop over all months
    for m in range (1,13):
        ind_b,ind_e=get_monthi(m)               #get index of beginning and end of month
        month=meter[ind_b:ind_e]

        # Check if months has numeric values        
        n_NaN=month.isnull().sum() #Number of NaN's
        if n_NaN<NaN_t:
            temps.append(get_mean_temp(row_wa,m))
            temps_min.append(get_mean_temp(row_wm,m))
            temps_max.append(get_mean_temp(row_wM,m))
            En_con.append(month.mean())


# In[100]:


nptemps=np.array(temps)
nptemps_min=np.array(temps_min)
nptemps_max=np.array(temps_max)
npEn_con=np.array(En_con)


# In[84]:


print(nptemps.shape)
print(nptemps)
print(npEn_con.shape)


# In[83]:


plt.scatter(nptemps,npEn_con)


# In[101]:


plt.scatter(nptemps_min,npEn_con)


# In[102]:


plt.scatter(nptemps_max,npEn_con)


# It's finally time to train the model <br>
# from : https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

# In[115]:


# Split the data into training/testing sets
X_train = nptemps[:-2000]
X_test = nptemps[-2000:]

X_train_min = nptemps_min[:-2000]
X_test_min = nptemps_min[-2000:]

X_train_max = nptemps_max[:-2000]
X_test_max = nptemps_max[-2000:]


# Split the targets into training/testing sets
y_train = npEn_con[:-2000]
y_test = npEn_con[-2000:]


# Create linear regression object
regr = linear_model.LinearRegression()
regr_min= linear_model.LinearRegression()
regr_max= linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)
regr_min.fit(X_train_min, y_train)
regr_max.fit(X_train_max, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
y_pred_min = regr_min.predict(X_test_min)
y_pred_max = regr_max.predict(X_test_max)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.8f'
      % mean_squared_error(y_test, y_pred))
print('Mean squared error min: %.8f'
      % mean_squared_error(y_test, y_pred_min))
print('Mean squared error max: %.8f'
      % mean_squared_error(y_test, y_pred_max))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


# In[91]:


# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[116]:


# Plot outputs
plt.scatter(X_test_min, y_test,  color='black')
plt.plot(X_test_min, y_pred_min, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[117]:


# Plot outputs
plt.scatter(X_test_max, y_test,  color='black')
plt.plot(X_test_max, y_pred_max, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# It could be that the prediction on the test set is better than a prediction on for a new data set because there are relativly few values for the temperature in the training/test data

# ## Combine temps for multi linear model

# In[112]:


combinetemps=[]
for i in range(len(temps)):
    combinetemps.append([temps[i], temps_min[i], temps_max[i]])
    
npcombinetemps=np.array(combinetemps)


# In[113]:


npcombinetemps.shape


# it's training time

# In[118]:


# Split the data into training/testing sets
X_train_comb = npcombinetemps[:-2000]
X_test_comb = npcombinetemps[-2000:]

# Split the targets into training/testing sets
y_train = npEn_con[:-2000]
y_test = npEn_con[-2000:]


# Create linear regression object
regr_comb = linear_model.LinearRegression()

# Train the model using the training sets
regr_comb.fit(X_train_comb, y_train)


# Make predictions using the testing set
y_pred_comb = regr_comb.predict(X_test)


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.8f'
      % mean_squared_error(y_test, y_pred_comb))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

