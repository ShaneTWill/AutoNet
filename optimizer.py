#!/usr/bin/env python

import sys
import os

# Dictionary of possible optimization methods.
optimizers = {
  0 : 'GradientDescent'
  ,1 : 'Adadelta'
  ,2 : 'Adagrad'
  ,3 : 'AdagradDA'
  ,4 : 'Momentum'
  ,5 : 'Adam'
  ,6 : 'Ftrl'
  ,7 : 'RMSProp'
}

# Dictionary of parameters and the associated keys to the example, invalid and conditions dictionaries.
parameters = {
  0: 'LearningRate,0'
  ,1: 'Momentum,1'
  ,2: 'Rho,0'
  ,3: 'Epsilon,0'
  ,4: 'InitialAccumulatorValue,0'
  ,5: 'InitialGradientSquaredAccumulatorValue,0'
  ,6: 'L1RegularizationStrength,1'
  ,7: 'L2RegularizationStrength,1'
  ,8: 'Beta1,1'
  ,9: 'Beta2,1'
  ,10: 'LearningRatePower,2'
  ,11: 'Decay,1'
}
		
# Dictionary of examples for the user to know what data to enter.
example = {
  0: 'a float greater than 0.0 ex. 0.05.'
  ,1: 'a float and must be greater than or equal to 0. i.e. 0.01.'
  ,2: 'a float and must be less than or equal to 0. i.e. -0.5.'
}

# Dictionary of messages to give the user on invalid data entry.
invalid = {
  0: 'not greater than zero'
  ,1: 'less than zero'
  ,2: 'greater than zero'
}

# Dictionary of lambda expressions for the conditional expressions.
conditions = {
  0: (lambda i: isinstance(float(i),float) and (float(i) > 0))
  ,1: (lambda i: isinstance(float(i),float) and (float(i) >= 0))
  ,2: (lambda i: isinstance(float(i),float) and (float(i) <= 0))
}

# Sets used for flow control when you choose to use the default parameters.
defaults = {
  0: {0,2,3,6,7}
  ,1: {1,5}
}

def getParameter(key):
  """
  Asks the user to select the value for a parameter.

  Keyword arguments:
  key -- An integer that corresponds to a key in the param dictonary.

  Returns:
  val -- The value selected by the user.
  """
  
  l = parameters[key].split(',')
  val = ''
  print('What value would you like for '+l[0]+'?')
  print('Please note that this is '+example[int(l[1])])
  i = sys.stdin.readline().strip()
  while True:
    if(conditions[int(l[1])](i)):
      val = str(i)
      break
        
    else:
      print('Sorry that was either  not a float or was '+invalid[int(l[1])]+', please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def OptimizerDefaultString(param):
  """
  Creates the code string for an optimizer if the default flag was set to true.

  Keyword arguments:
  param -- A python list that contains the optimizer key and parameter settings.

  Returns:
  optimizer -- The optimizer code as a string.
  """

  data = param[1][0]
  optimizer = []
  optimizer.append('optimizer = tf.train.{0}Optimizer('.format(optimizers[data]))
  if(data in defaults[0]):
    optimizer.append(')')
  
  elif(data in defaults[1]):
    optimizer.append('learning_rate={0})'.format(param[1][2]))
  
  else:
    optimizer.append('learning_rate={0}, momentum={1})'.format(param[1][2],param[1][3]))
  
  return ''.join(optimizer)


def OptimizerString(param):
  """
  Creates the code for a non-default optimizer string and returns it.

  Keyword arguments:
  param -- a python list that contains the optimizer key and parameter settings.

  Returns:
  optimizer -- The optimizer code as a string. 
  """

  data = param[1][0]
  optimizer = []
  optimizer.append('optimizer = tf.train.{0}'.format(optimizers[data]))
  optimizer.append('Optimizer(learning_rate={0}'.format(param[1][2]))
  
  if(data == 0):
    optimizer.append(')\n')
      
  elif(data == 1):
    optimizer.append(', rho={0}'.format(param[1][3]))
    optimizer.append(', epsilon={0})\n'.format(param[1][4]))
      
  elif(data == 2):
    optimizer.append(', initial_accumulator_value={0})\n'.format(param[1][3]))
      
  elif(data == 3):
    optimizer.append(', initial_gradient_squared_accumulator_value={0}'.format(param[1][3]))
    optimizer.append(', l1_regularization_strength={0}'.format(param[1][4]))
    optimizer.append(', l2_regularization_strength={0})\n'.format(param[1][5]))
      
  elif(data == 4):
    optimizer.append(', momentum={0})\n'.format(param[1][3]))
      
  elif(data == 5):
    optimizer.append(', beta1={0}'.format(param[1][3]))
    optimizer.append(', beta2={0}'.format(param[1][4]))
    optimizer.append(', epsilon={0})\n'.format(param[1][5]))
      
  elif(data == 6):
    optimizer.append(', learning_rate_power={0}'.format(param[1][3]))
    optimizer.append(', initial_accumulator_value={0}'.format(param[1][4]))
    optimizer.append(', l1_regularization_strength={0}'.format(param[1][5]))
    optimizer.append(', l2_regularization_strength={0})\n'.format(param[1][6]))
      
  else:
    optimizer.append(', decay={0}'.format(param[1][3]))
    optimizer.append(', momentum={0}'.format(param[1][4]))
    optimizer.append(', epsilon={0}'.format(param[1][5]))

  return ''.join(optimizer)


def getOptimizer():
  """
  Gets the optimizer settings and returns a string that is a list of optimizer settings.

  This method gets all the information needed for the optimizer. These values are then
  used to construct a python list with each position in the list corresponding to a set
  of values for the optimizer.
  
  Returns:
  ary -- A string that is a valid python list containing the optimizer key and parameter settings.
  """

  ary = '['
  cns = ''
  df = 0
  key = 0

  ary , key = ChooseOptimizer(ary)
  ary , df = SetDefault(ary)

  if(df==1):
    ary = ary + getOptimizerDefaultParameters(key)
      
  else:
    ary = ary + getOptimizerParameters(key)
    
  ary = ary + ']'     
  os.system('cls' if os.name == 'nt' else 'clear')
  
  return ary


def ChooseOptimizer(ary):
  """
  Asks the user to choose an optimizer from the dictionary of choices.
  
  Keyword arguments:
  ary --  A string used to create the list of the optimizer parameters.

  Returns:
  ary -- The modified string ary.
  key -- The chosen optimizer key as an int. 
  """

  while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print('What Optimizer from the list below do you want to optimize with?')
    for key,value in optimizers.items():
      print('{0}: {1}'.format(key,value))

    i = sys.stdin.readline().strip()
    if(i.isdigit() and (-1<int(i)<8)):
      key = int(i)
      ary = ary + str(i) + ','
      break
  
    else:
      print('Sorry that was not an integer, or an option, please try again.')

  os.system('cls' if os.name == 'nt' else 'clear')

  return ary,key


def SetDefault(ary):
  """
  Asks the user for the value of the default key in the optimizer parameter list.

  Keyword arguments:
  ary -- A string that will be modified to include in default parameter.

  Returns:
  ary -- The modified version of the input string ary.
  df -- A binary value to signify the selection of a default optimizer.
  """

  df = 0
  print('Would you like to use the default settings or not?\nDefault: 1\tNot: 0.')
  i = sys.stdin.readline().strip()
  while True:
    if(i.isdigit() and (-1<int(i)<2)):
      df = int(i)
      ary = ary + str(i) + ','
      break
  
    else:
      print('Sorry that was not an integer, or an option, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return ary,df


def getOptimizerDefaultParameters(key):
  """
  Calls all methods needed for optimizers with only default parameters.

  Keyword arguments:
  key -- The optimizer key.

  Returns:
  s -- String of optimizer settings as comma separated values.
  """

  s = ''
  if(key in defaults[1]):
    s = ''

  elif(key in defaults[0]):
    l = getParameter(0)
    s = l

  else:
    l = getParameter(0)
    m = getParameter(1)
    s = l + ',' + m

  return s


def getOptimizerParameters(key):
  """
  Calls the parameter selection methods needed for each optimizer type and returns a string of comma separated
  values for each optimizer.

  Keyword arguments:
  key -- The optimizer of choice.

  Returns:
  s -- The string of optimizer parameter values as a comma separated string.
  """

  s = ''
  l = getParameter(0)
  if(key==0):
    s = l

  elif(key==1):
    r = getParameter(2)
    e = getParameter(3)
    s = l + ',' + r + ',' + e

  elif(key==2):
    initial = getParameter(4)
    s = l + ',' + initial

  elif(key==3):
    gsa = getParameter(5)
    l1 = getParameter(6)
    l2 = getParameter(7)
    s = l + ',' + gsa + ',' + l1 + ',' + l2

  elif(key==4):
    m = getParameter(1)
    s = l + ',' + m

  elif(key==5):
    b1 = getParameter(8)
    b2 = getParameter(9)
    e = getParameter(3)
    s = l + ',' + b1 + ',' + b2 + ',' + e

  elif(key==6):
    lr = getParameter(10)
    a = getParameter(4)
    l1 = getParameter(6)
    l2 = getParameter(7)
    s = l + ',' + lr + ',' + a + ',' + l1 + ',' + l2

  else:
    d = getParameter(11)
    m = getParameter(1)
    e = getParameter(3)
    s = l + ',' + d + ',' + m + ',' + e

  return s


