#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:09:46 2023

@author: tara
"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv('RandomDipole(T50000).csv')

# Create 3D scatter plot with original data
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
graph1 = ax1.scatter(df['xd'], df['yd'], df['zd'], c=df['AC_SBz_lower'])
ax1.set_xlabel('xd')
ax1.set_ylabel('yd')
ax1.set_zlabel('zd')
ax1.set_title('Original Data 3D Scatter Plot')
plt.colorbar(graph1, ax=ax1, pad=0.1)
plt.show()

# Create 2D scatter plot with original data
fig = plt.figure()
ax2 = fig.add_subplot(111)
graph2 = ax2.scatter(df['BC_SBz_lower'], df['AC_SBz_lower'], s=0.1)
ax2.set_xlabel('BC_SBz_lower')
ax2.set_ylabel('AC_SBz_lower')
ax2.set_title('Original Data 2D Scatter Plot')
plt.show()

# Filter the dataframe to only include rows where 'AC_DBz_lower' > 0.25
df_filtered = df[df['AC_SBz_lower'] > 0.25]

# Create 3D scatter plot with filtered data
fig = plt.figure()
ax3 = fig.add_subplot(111, projection='3d')
graph3 = ax3.scatter(df_filtered['xd'], df_filtered['yd'], df_filtered['zd'], c=df_filtered['AC_SBz_lower'])
ax3.set_xlabel('xd')
ax3.set_ylabel('yd')
ax3.set_zlabel('zd')
ax3.set_title('Filtered Data 3D Scatter Plot')
plt.colorbar(graph3, ax=ax3, pad=0.1)
plt.show()
