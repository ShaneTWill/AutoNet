#!/usr/bin/env python

import sys
import os


# Dictionary of possible cost functions.
met = {
        0 : 'RMSE: Root Mean Squared Error',
        1 : 'MSE: Mean Squared Error',
        2 : 'MAE: Mean Absolute Error',
        3 : 'MAPE: Mean Absolute Percent Error'
        }


def getMetric():
  """
  Calls the methods to choose the cost metric and the number of epochs to train over.
  
  Once the cost metric and number of epochs have been chosen a string is returned.
  This string takes the form of a python list with the data stored in the following
  format: ['metric','epochs']

  Returns:
  A string that is valid python code for the creation of a list.

  Example: "[0,500]"
  """

  ary = '['
  e = ']'
  
  ary = ChooseMetric(ary)
  ary = getEpoch(ary)
  ary = ary + e

  return ary


def ChooseMetric(ary):
  """
  Asks the user to choose a cost function to optimize by, and returns the value chosen. 

  Keyword arguments:
  ary --  A string that the metric is concatenated on to.

  Returns:
  ary -- The inputted string with the metric concatenated onto it.
  """
  
  print('What metric from the list below do you want to optimize by?')
  for k in met.keys():
    print(str(k)+": "+str(met[k]))

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


def getEpoch(ary):
  """
  Gets the number of epochs to train over from the user and returns the modified 
  string ary.

  Keyword arguments:
  ary -- A string that the number of epochs is concatenated on to.

  Returns:
  ary -- The inputted string with number of epochs concatenated onto it.
  """

  print('Finally how many epochs should the model be trained over?')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit() and (int(i)>0)):
      ary = ary + str(i)
      break
  
    else:
      print('Sorry was not an integer.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')

  return ary


