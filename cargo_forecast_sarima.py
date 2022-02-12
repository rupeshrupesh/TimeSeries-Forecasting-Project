#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:32:47 2020

@author: rupeshr
"""
print(range(0,2))
for i in range(0,1):
    print(i)
import pandas as pd
import numpy as np
import warnings
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sm
from sklearn.metrics import mean_squared_error
from datetime import datetime as dat, timedelta
from pandas import datetime
from dateutil.relativedelta import relativedelta
import datetime
from datetime import date
from pandas import Series
from math import sqrt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from rest_framework.response import Response
from rest_framework.views import APIView
from cargo.cargomodules.models import Forecasttimeseriesdetails
from django_pandas.io import read_frame
from django.db import connection

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

def inverse_difference(last_ob, value):
	return value + last_ob


def adfuller_test(values):
    result = adfuller(values)
    try:
        if result[1] <= 0.05:
            return pd.DataFrame(values)
        else:
            diff = difference(values)
            inverted = [inverse_difference(values[i], diff[i]) for i in range(len(diff))]
            inverted = pd.DataFrame(inverted)
            return inverted
    except ValueError:
        print("Oops!The matrix was incorrect.Try again...")

def evaluate_forecast_sarima(data):
    df=pd.DataFrame(data,columns=['week','year','originPort','deliveryPort','bound','twentys','fortys'])
    df['date'] = pd.to_datetime(df.year.astype(str), format='%Y') + \
                   pd.to_timedelta(df.week.mul(7).astype(str) + ' days')
    df['portPair'] = df['originPort']+"-"+df['deliveryPort']+"-"+df['bound']
    
    equipment={'20':'twentys','40':'fortys'}
    portpair = df['portPair'].unique()
    #try:
    #    portdata=df[(df.portPair==portpair)]
    #except ValueError:
     #   print("The given portpair is wrong")
    #else:
      #  print("The given portpair is correct")
    
    for i in portpair:
        portdata=df[(df.portPair==i)]
        for j in equipment:
            datalist = ['date']
            datalist.append(equipment[j])
            forecastdata = portdata[datalist]
    #portdata = portdata.sort_values(by='date')
            forecastdata=forecastdata.set_index('date')
            mindate=forecastdata.index.min()
            maxdate=forecastdata.index.max()
            warnings.filterwarnings("ignore")
            data=evaluate_models(forecastdata.values,i,j,mindate,maxdate)
    return data
    
def evaluate_models(dataset,portpair,equipmenttype,datefrom,dateto):
    now=datetime.datetime.now()
    if len(dataset) >=5:
        try:
            valueddata = adfuller_test(dataset)
            dataset = valueddata.values
            dataset = dataset.astype('float32')
            best_cfg=(1,1,0)
            best_seasonal_order=(1,0,1,52)
            model = SARIMAX(dataset, order=best_cfg,seasonal_order=best_seasonal_order,enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()[0]
            print('output:',output.astype('int'),'portpair:',portpair,'equipment:',equipmenttype,'datefrom:',datefrom,'dateto:',dateto)
            return output
            #portpairs=portpair.split("-")
            #service=portpairs[0]
            #originport=portpairs[1]
            #deliveryport=portpairs[2]
            #bound=portpairs[3]
            #curryear=dat.now().year
            #week =date.today().isocalendar()[1]
            
            
            #Forecasttimeseriesdetails(portpair=portpair, equipmenttype=equipmenttype, best_arima=str(best_cfg),best_sarima=str(best_seasonal_order)\
             #            ,originport=originport,deliveryport=deliveryport,\
            #mse="mse", datafrom=datefrom, datato=dateto,prediction=output,forecasteddate=now,week=week,year=curryear,bound=bound).save()
            #print("savedsuccessfully")
            #return output
        except Exception as LinAlgError:
            print(LinAlgError)
    else:
        print('Insufficient data to build forecasting model')
        

import pymssql
conn = pymssql.connect(server='192.168.10.89',port='62775',database='CARGOOPS',user='SVMRead',password='$vM@Kan00Dev')
cursor = conn.cursor()
cursor.execute("select servicecode,vesselcode,voyage,bound,status,oprtype,containerstatus,soc,mode,actdate,originPort,deliveryPort,datepart(year,actdate) as year,datepart(week,actdate) as week, equipmenttype,\
        (case when equipmenttype like '%2%' then 1 else 0 end) as twenty,(case when equipmenttype like '%4%' then 1 else 0 end) as forty into #cargo from operationdetail o \
        where o.status='L' and o.oprtype='export' and (o.containerstatus='F' or (o.containerstatus='M' and o.soc='Y')) and o.mode='M' and actdate>='2016-01-01' and actdate<'2021-01-01'; \
        select  week,year,originPort,deliveryPort,bound,sum(twenty) as twentys,sum(forty) as fortys from #cargo \
        group by originPort,deliveryPort,bound,year,week;")
data = cursor.fetchall()
evaluate_forecast_sarima(data)



import pymysql
conn = pymysql.connect(host='localhost',port=3306,database='outputdb',user='rupesh',password='P@ssw0rd')
cursor = conn.cursor()
cursor.execute()



import pymysql
import pandas as pd
conn = pymysql.connect(host='localhost',port=3306,database='outputdb',user='rupesh',password='P@ssw0rd')
cursor = conn.cursor()
sql=pd.read_sql_query('select * from sample1 limit 10',conn)
print(sql)



def function(a,b,c):
    try:
        if a>b:
            print("a is greater than b")
        else:

x=5            
try:
  print(x)
except NameError:
  print("Variable x is not defined")
except:
  print("Something else went wrong")
  

  
try:
  print(x)
except:
  print("Something went wrong")
finally:
  print("The 'try except' is finished")
  
  
  
  
def function(a,b):
    try:
        if a>b:
            print("a is greater than b")
    except:
        print("please specify the numbers instead of characters")
    return

function(3,4)



for x in range(6,20):
    if(x%2==0):
        print(x)
        
n=10
fact=1
for x in range(1,n+1):
    fact = fact * x
    print(fact)
    
num= int(input("enter the number:"))
i=0
n1=0
n2=0
while(i<num):
    if(i<=1):
        sum=i
    else:
        sum=n1+n2
        n1=n2
        n2=sum
    print(sum, end="")
    i=i+1
    

f = open("/home/rupeshr/Desktop/demofile.txt", "r")
print(f.read())
print(f.readline())
f.close()

f = open("/home/rupeshr/Desktop/demofile.txt", "a")
f.write("Now the file has more content!")
f.close()

f = open("/home/rupeshr/Desktop/demofile.txt", "w")
f.write("Woops! I have deleted the content!")
f.close()

import os
os.remove("/home/rupeshr/Desktop/demofile.txt")









