#!/usr/bin/env python


import os 
import sys


# Dictionary of possible activation functions.
activations = {
               0 : 'relu'
               ,1 : 'relu6'
               ,2 : 'crelu'
               ,3 : 'elu'
               ,4 : 'softplus'
               ,5 : 'softsign'
              }

# Dictionary of input and output prompts
prompts = {
           0 : 'How many inputs are there?,0,0'
           ,1 : 'How many outputs are there?\nIf this is a classification model that uses one-hot encoding enter number of possibile values.,0,0'
           ,2 : 'First thing is first: enter \'R\' for a regression model and \'C\' for a classification problem.,1,1'
           ,3 : 'Now how many hidden layers are there?,2,2'
          }


conditions = {
              0 : (lambda i: i.isdigit() and int(i) > 0)
              ,1 : (lambda i: i in {'C','c','R','r'})
              ,2 : (lambda i: i.isdigit() and int(i) > -1)
             } 


errors = {
          0 : 'Sorry that was either not a number or less than 1. try again'
          ,1 : 'Sorry that is not an option try again.'
          ,2 : 'Sorry that was either not an integer or less than zero.'
         }


def _getData(key):
  """
    A general method that uses gets and data from the user. The model takes an integer as a parameter. This 
    intger then pulls the correct prompt from the prompts dictionary and parses the control flow meta data 
    from the prompt. 

    Keyword Inputs: 
    
    key - the integer key for the prompt desired from the prompts dictionary.

    Returns:

    data - the information that the user has inputed into the system.

  """
  data = ''
  prompt , condKey, err = prompts[key].split(',')
  condKey = int(condKey)
  err = int(err)

  while True:
    print(prompt)
    i = sys.stdin.readline().strip()
    if(conditions[condKey](i)):
      data = i
      break

    else:
      print(errors[err])

  os.system('cls' if os.name == 'nt' else 'clear')

  return data

def getModelBasics():
  """
  Calls methods to obtain the basics of a predictive model:
  model type, number of inputs, and number of outputs. 

  Returns:
  A list of model parameters and the model type.
  """

  t = '\'float64\''
  n = 0
  o = 0
  m = ''
  m = _getData(2)
  n = _getData(0)
  o = _getData(1)
 
  return [[t,n]],[[t,o]],m


def getHiddenLayers():
  """
  Runs the methods for getting the number of hidden layers, and the number of nodes, and 
  activation function for each layer.

  Returns:
  e -- A python list of lists with each sub-list containing the parameters for a layer in the model.
  """

  ary = '['
  cnt = 0
  cns = ''
  
  i = _getData(3)
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


def getNumberofNodes(cnt,cns):
  """
  Asks the user for the number of outputs in the model.

  Keyword arguments:
  cnt -- The layer number.
  cns -- String of layer parameters that will be used to create a python list.

  Returns:
  cns -- A string that represents part of a python list with the number of nodes.
  """

  n = 0
  print('For layer '+str(cnt+1)+'.\nHow many nodes are there?')
  while True:
    n = sys.stdin.readline().strip()
    if(n.isdigit() and int(n) > 0):
      cns = '['+str(n)+','
      break
        
    else:
      print('Sorry that was either not an integer, or less than 1.')
      n = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
            
  return cns


def getActivationFunction(cnt,cns):
  """
  Asks the user for the number of outputs in the model.

  Keyword arguments:
  cnt -- The layer number.
  cns -- String of layer parameters that will be used to create a python list.

  Returns:
  cns -- The data type for the outputs for the model.
  """

  n = 0
  print('For layer '+str(cnt+1)+'.')
  print('Below is a list of the activation functions supported. Choose one.') 
  for key,value in activations.items():
      print('{}: {}'.format(key,value))

  while True:
    n = sys.stdin.readline().strip()
    if((n.isdigit()) and (-1 < int(n) < 6)):
      cns = cns + str(n) + ']'
      break
        
    else:
      print('Sorry that was not an option please try again.')
  os.system('cls' if os.name == 'nt' else 'clear')
    
  return cns


