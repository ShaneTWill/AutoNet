#!/usr/bin/env python

#=========================================================================#
#									  #
# This file holds all the code for choosing the cost function settings,   #
# and the number of epochs 
# 									  #
#=========================================================================#

import sys
import os


# Dictionary of Possible metrics
met = {
        0 : 'RMSE',
        1 : 'MSE',
        2 : 'MAE',
        3 : 'MAPE'
        }


# Asks for the input of data for cost metric, and the number of epochs to train over.
def getMetric():
  ary = '['
  e = ']'
  
  ary = ChooseMetric(ary)
  ary = getEpoch(ary)
  ary = ary + e

  return ary


# Asks the user for the metric to optimize by.
def ChooseMetric(ary):
  print('What metric from the list do you want to optimize by?\n'+str(met))
  print('Please enter the integer corresponding to the correct metric.')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit() and (-1<int(i)<4)):
      ary = ary + str(i) + ','
      break
    
    else:
      print('Sorry that was not an integer, or an option, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return ary


# Asks the user for the number of iterations to train over.
def getEpoch(ary):
  print('Finally how many epochs should the model be trained over?')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      ary = ary + str(i)
      break
  
    else:
      print('Sorry was not an integer.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')

  return ary


