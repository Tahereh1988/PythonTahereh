#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:09:46 2023

@author: tara
"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


df = pd.read_csv('RandomDipole(T50000).csv')

# Create 3D scatter plot with original data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
graph1 = ax.scatter(df['xd'], df['yd'], df['zd'], c=df['AC_DBz_lower'])
# ax.hist(df['AC_SBz_lower'],100)
# create 2D acatter plot with original data
# graph3 = ax.scatter(df['BC_DBz_lower'], df['AC_DBz_lower'], s=.1)
# ax.set_xlabel('BC_DBz_lower')
# ax.set_ylabel('AC_DBz_lower')
ax.set_xlabel('xd')
ax.set_ylabel('yd')
ax.set_zlabel('zd')
plt.title('After-Corection-DeltaBz-lower(nT)')
plt.colorbar(graph1, pad=0.15)
plt.show()

# Filter the dataframe to only include rows where 'xd' ,' yd' and 'zd' are within certain range
df_filtered = df[(df['AC_DBz_lower'] > 0.25)]
# df_filtered = df
# print(df_filtered)

# Create 3D scatter plot with filtered data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
graph2=ax.scatter(df_filtered['xd'], df_filtered['yd'], df_filtered['zd'], c=df_filtered['AC_DBz_lower'])
# graph2=ax.scatter(df_filtered['BC_DBz_upper'],df_filtered['AC_DBz_upper'],s=.1)
# ax.axhline(y=0.2, color='r')
# graph4=ax.hist(df_filtered['BC_SBz_upper'], df_filtered['AC_SBz_upper'])
ax.set_xlabel('xd')
ax.set_ylabel('yd')
ax.set_zlabel('zd')
plt.title('Filtered-After-Corection-DeltaBz-lower(nT)')
plt.colorbar(graph2, pad=0.15)
plt.show()