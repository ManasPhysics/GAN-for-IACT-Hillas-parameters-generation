#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:51:34 2023

@author: manas
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math 
import pandas as pd 
import seaborn as sns



#### Reading Real(simulation data from file ###################
columns=['Energy','Length', 'Width', 'Dist', 'Alpha', 'F2', 'Size','Asym']
data=pd.read_csv("HillasGamma_Zen05_PE18.dat",sep='\s+',usecols=columns)

data=data[(data.Alpha<89.0) & (data.Alpha>0.0)]
data.Alpha=-np.log(3.14*data.Alpha/180.0) 
data.Energy=np.log10(data.Energy)
data.Size=np.log10(data.Size)
# Normalize the input data(i.e. mean=0,std=1.0) 
data=data[data.Asym>0.0]   ### assending order of size 
#data = data.('Asym', ascending=True)
data=(data-data.mean(axis=0))/np.std(data)

### Renaming the names of the columns ####################################################################
data = data.rename({'Energy': 'log10(Energy)', 'Size': 'log10(Size)' , 'Alpha':'log(Alpha)'}, axis=1)

## Y is the Pandas dataframe of X(generated variables)
Y=0.0
Y=pd.DataFrame(X,columns=data.columns)
Y=(Y-Y.mean()/(Y.std()))


######## Adding labels to the dataframe ##############################################################
data['Outcome']='Simulation' # simulated data is labelled as 1.0 
Y['Outcome']='GAN'  # 



#### Sampling data for PairPlots #############
 NS=10000
 RS_D=data.sample(NS)
 RS_S=Y.sample(NS)
 RS=pd.concat([RS_D,RS_S],ignore_index=True)


## Axes labelling font size changeing ###################
sns.set( rc = {'axes.labelsize' : 12. })
sns.set( rc = {'axes.labelweight': 'bold'})


#### Fill = False command prevents diagonal plot to appear
P_Plot=sns.pairplot(RS, hue ='Outcome', corner='True' , markers='.', palette=['red','blue'],plot_kws={"s": 10}, diag_kind='hist', diag_kws={"linewidth":0.9 , "fill": False})
P_Plot.tick_params(labelsize=9.0) ### Axis labelling tick size ==0.0 


