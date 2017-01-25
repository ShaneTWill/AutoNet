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


# Gets the data on number of inputs, outputs, and model type.
def getModelBasics():
  t = ''
  n = 0
  o = 0
  m = ''
  t , m = getModelType()
  n = getInputCount()
  o = getOutputCount()
 
  return [[t,n]],[[t,o]],m


# Asks the user for the number of inputs.
def getInputCount():
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

  
# Asks the user for the number of outputs.
def getOutputCount():
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


# Asks the user for the model type.
def getModelType():
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


# Asks the user for the data on the hidden layers.
def getHiddenLayers(out):
  ary = '['
  cnt = 0
  cns = ''
  
  i = getNumberOfLayers(cnt)
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


# Asks the user for the number of layers in the neural network.
def getNumberOfLayers(cnt):
  print('Now how many hidden layers  are there?')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      break
    else:
      print('Sorry that was not an integer.')
  os.system('cls' if os.name == 'nt' else 'clear')

  return i


# Asks the user how many nodes are in a layer.
def getNumberofNodes(cnt,cns):
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


# Asks the user to choose an activation function for a layer.
def getActivationFunction(cnt,cns):
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


