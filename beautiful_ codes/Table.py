#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:14:51 2023

@author: tara
"""

# save the table in a variable
from tabulate import tabulate
table = tabulate({'l': [1, 1, 2, 2, 2, 3, 3, 4], 
                 'm': [0, 1, 0, 1, 2, 1, 3, 2], 
                 'Delta(B_z) upper & lower before': [],
                 'Delta(B_z) upper after':[],
                 'Delta(B_z) lower after':[],
                 'Sigma(B_z) upper & lower before':[],
                 'Sigma(B_z) upper after':[],
                 'Sigma(B_z) lower after':[],
                 '<BT> before': [],
                 '<BT> before': []},
                 headers='keys', 
                 tablefmt='fancy_grid', 
                 )

# use context manager to create table.txt file and write table to it
with open('table3.txt', 'w') as f:
  f.write(table)