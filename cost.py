#!/usr/bin/env python

import sys
import os


# Dictionary of possible cost functions.
metrics = {
           0 : 'RMSE: Root Mean Squared Error'
           ,1 : 'MSE: Mean Squared Error'
           ,2 : 'MAE: Mean Absolute Error'
           ,3 : 'MAPE: Mean Absolute Percent Error'
          }

# Dictionary of conditional boolean statements used. 
conditions = {
              0: (lambda i: i.isdigit() and (-1<int(i)<len(metrics)))
              ,1: (lambda i: i.isdigit() and (int(i)>0))
             }

# Dictionary of error message strings.
errors = {
          0: 'Sorry that was not an integer, or an option, please try again.'
          ,1: 'Sorry was not an integer.'
         }

# Dictionary of prompt message strings.
prompts = {
           0: 'What metric from the list below do you want to optimize by?'
           ,1: 'Finally how many epochs should the model be trained over?'
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
  
  ary = getData(0,ary)
  ary = getData(1,ary)
  ary += e

  return ary


def getData(code,ary):
  """
    Uses data that the user inputs and and returns the modified 
    string ary. For the creation of a list of values.

    Keyword arguments:
    code -- An integer that is used to pull the correct data from the dictionaries 
            and perform flow control.
    ary -- A string that the number of epochs is concatenated on to.

    Returns:
    ary -- The inputted string with the data concatenated onto it.
  """
  
  print(prompts[code])
  if(code == 0):
    for key, value in metrics.items():
      print('{}: {}'.format(key,value))

  while True:
    i = sys.stdin.readline().strip()
    if(conditions[code](i)):
      ary = ary + str(i) + ','
      break
                                          
    else:
      print(errors[code])
  os.system('cls' if os.name == 'nt' else 'clear')

  return ary

