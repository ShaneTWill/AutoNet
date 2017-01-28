#!/usr/bin/env python


import os 
import sys


# Dictionary of possible activation functions.
act = {
        0 : 'relu',
        1 : 'relu6',
        2 : 'crelu',
        3 : 'elu',
        4 : 'softplus',
        5 : 'softsign'
        }


def getModelBasics():
  """
  Calls methods to obtain the basics of a predictive model:
  model type, number of inputs, and number of outputs. 

  Returns:
  A list of model parameters and the model type.
  """

  t = ''
  n = 0
  o = 0
  m = ''
  t , m = getModelType()
  n = getInputCount()
  o = getOutputCount()
 
  return [[t,n]],[[t,o]],m


def getInputCount():
  """
  Asks the user for the number of inputs in the model.

  Returns:
  n -- The number of inputs selected by the user.
  """

  n = ''
  print('Now how many inputs are there?\nPlease enter an integer like 1.')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      n = i
      break
        
    else:
      print('Sorry that is not an option try again.')

  os.system('cls' if os.name == 'nt' else 'clear')
      
  return n


def getOutputCount():
  """
  Asks the user for the number of outputs in the model.

  Returns:
  o -- The number of outputs selected by the user.
  """

  o = ''
  print('How many outputs are there?')
  print('If this is a classification model that uses one-hot encoding enter number of possibile values.')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      o = i
      break
        
    else:
      print('Sorry that is not an option try again.')

  os.system('cls' if os.name == 'nt' else 'clear')

  return o


def getModelType():
  """
  Asks the user for the number of outputs in the model.

  Returns:
  t -- The data type for the outputs for the model.
  m -- The model type selected by the user.
  """

  t = ''
  m = ''
  print('First thing is first: enter \'R\' for a regression model, and \'C\' for a classification problem.')
  while True:
    i = sys.stdin.readline().strip()
    if(i=='R'):
      t = '\'float64\''
      m = i
      break
    
    elif(i=='C'):
      t = '\'float64\''
      m = i
      break
        
    else:
      print('Sorry that is not an option try again.')

  os.system('cls' if os.name == 'nt' else 'clear')
      
  return t,m


def getHiddenLayers():
  """
  Runs the methods for getting the number of hidden layers, and the number of nodes, and 
  activation function for each layer.

  Returns:
  e -- A python list of lists with each sub-list containing a parameters for a layer in the model.
  """

  ary = '['
  cnt = 0
  cns = ''
  
  i = getNumberOfLayers()
  while(cnt < int(i)):
    cns = getNumberofNodes(cnt,cns)
    cns = getActivationFunction(cnt,cns)
    if((cnt+1) != int(int(i))):
      ary += cns + ','
      cns = ''

    else:
      ary += cns + ']'
      
    cnt += 1
  
  if(int(i)==0):
    ary = ary + ']'

  e = eval(ary)
    
  return e


def getNumberOfLayers():
  """
  Asks the user for the number of layers in the model.

  Returns:
  i -- The data type for the number of layers in the model.
  """

  print('Now how many hidden layers  are there?')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      break
    else:
      print('Sorry that was not an integer.')
  os.system('cls' if os.name == 'nt' else 'clear')

  return i


def getNumberofNodes(cnt,cns):
  """
  Asks the user for the number of outputs in the model.

  Keyword arguments:
  cnt -- The layer number.
  cns -- String of layer parameters for formatted to create a python list.

  Returns:
  cns -- A string that represents part of a python list with the number of nodes.
  """

  n = 0
  print('For layer '+str(cnt+1)+'.\nHow many nodes are there?')
  while True:
    n = sys.stdin.readline().strip()
    if(n.isdigit()):
      cns = '['+str(n)+','
      break
        
    else:
      print('Sorry that was either not an integer, or an option, please try again.')
      n = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
            
  return cns


def getActivationFunction(cnt,cns):
  """
  Asks the user for the number of outputs in the model.

  Keyword arguments:
  cnt -- The layer number.
  cns -- String of layer parameters for formatted to create a python list.

  Returns:
  cns -- The data type for the outputs for the model.
  """

  n = 0
  print('For layer '+str(cnt+1)+'.\n'+str(act))
  print('That list is the activation functions supported what is this layer made of?')
  while True:
    n = sys.stdin.readline().strip()
    if((n.isdigit()) and (-1 < int(n) < 6)):
      cns = cns + str(n) + ']'
      break
        
    else:
      print('Sorry that was not an integer please try again.')
  os.system('cls' if os.name == 'nt' else 'clear')
    
  return cns


