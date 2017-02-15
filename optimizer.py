#!/usr/bin/env python

import sys
import os


# Dictionary of possible optimization methods.
opt = {
  0 : 'GradientDescent',
  1 : 'Adadelta',
  2 : 'Adagrad',
  3 : 'AdagradDA',
  4 : 'Momentum',
  5 : 'Adam',
  6 : 'Ftrl',
  7 : 'RMSProp'
}

# Dictionary of parameters and the associated keys to the example, invalid and conditions dictionaries.
params = {
  0: 'LearningRate,0',
  1: 'Momentum,1',
  2: 'Rho,0',
  3: 'Epsilon,0',
  4: 'InitialAccumulatorValue,0',
  5: 'InitialGradientSquaredAccumulatorValue,0',
  6: 'L1RegularizationStrength,1',
  7: 'L2RegularizationStrength,1',
  8: 'Beta1,1',
  9: 'Beta2,1',
  10: 'LearningRatePower,2',
  11: 'Decay,1'
}
		
# Dictionary of examples for the user to know what data to enter.
example = {
  0: 'a float greater than 0.0 ex. 0.05.',
  1: 'a float and must be greater than or equal to 0. i.e. 0.01.',
  2: 'a float and must be less than or equal to 0. i.e. -0.5.'
}

# Dictionary of messages to give the user on invalid data entry.
invalid = {
  0: 'not greater than zero',
  1: 'less than zero',
  2: 'greater than zero'
}

# Dictionary of lambda expressions for the conditional expressions.
conditions = {
  0: (lambda i: isinstance(float(i),float) and (float(i) > 0)),
  1: (lambda i: isinstance(float(i),float) and (float(i) >= 0)),
  2: (lambda i: isinstance(float(i),float) and (float(i) <= 0))
}
			

def getParameter(key):
  """
  Asks the user to select the value for a parameter.

  Keyword arguments:
  key -- An integer that corresponds to a key in the param dictonary.

  Returns:
  val -- The value selected by the user.
  """
  
  l = params[key].split(',')
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
  o -- The optimizer code as a string.
  """

  o = '  '
  if(param[1][0]==0):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])+')\n'

  elif(param[1][0]==1):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer()\n'
      
  elif(param[1][0]==2):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])+')\n'
      
  elif(param[1][0]==3):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])+')\n'
      
  elif(param[1][0]==4):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', momentum='+str(param[1][3])+')\n'
      
  elif(param[1][0]==5):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer()\n'
      
  elif(param[1][0]==6):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])+')\n'
      
  else:
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])+')\n'
    
  return o


def OptimizerString(param):
  """
  Creates the code for a non-default optimizer string and returns it.

  Keyword arguments:
  param -- a python list that contains the optimizer key and parameter settings.

  Returns:
  o -- The optimizer code as a string. 
  """

  o = '  '
  if(param[1][0]==0):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])+')\n'
      
  elif(param[1][0]==1):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', rho='+str(param[1][3])
    o = o + ', epsilon='+str(param[1][4])+')\n'
      
  elif(param[1][0]==2):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', initial_accumulator_value='+str(param[1][3])+')\n'
      
  elif(param[1][0]==3):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', initial_gradient_squared_accumulator_value='+str(param[1][3])
    o = o + ', l1_regularization_strength='+str(param[1][4])
    o = o + ', l2_regularization_strength='+str(param[1][5])+')\n'
      
  elif(param[1][0]==4):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', momentum='+str(param[1][3])+')\n'
      
  elif(param[1][0]==5):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', beta1='+str(param[1][3])
    o = o + ', beta2='+str(param[1][4])
    o = o + ', epsilon='+str(param[1][5])+')\n'
      
  elif(param[1][0]==6):
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', learning_rate_power='+str(param[1][3])
    o = o + ', initial_accumulator_value='+str(param[1][4])
    o = o + ', l1_regularization_strength='+str(param[1][5])
    o = o + ', l2_regularization_strength='+str(param[1][6])+')\n'
      
  else:
    o = o + 'optimizer = tf.train.'+opt[param[1][0]]
    o = o + 'Optimizer(learning_rate='+str(param[1][2])
    o = o + ', decay='+str(param[1][3])
    o = o + ', momentum='+str(param[1][4])
    o = o + ', epsilon='+str(param[1][5])+')\n'
    
  return o


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
    for k in opt.keys():
      print(str(k)+": "+str(opt[k]))

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


# Gets the optimizer values needed if default is chosen.
def getOptimizerDefaultParameters(key):
  """
  Calls all methods needed for optimizers with only default parameters.

  Keyword arguments:
  key -- The optimizer key.

  Returns:
  s -- String of optimizer settings as comma separated values.
  """

  s = ''
  if(key==0):
    l = getParameter(0)
    s = l

  elif(key==1):
    s = ''

  elif(key==2):
    l = getParameter(0)
    s = l

  elif(key==3):
    l = getParameter(0)
    s = l

  elif(key==4):
    l = getParameter(0)
    m = getParameter(1)
    s = l + ',' + m

  elif(key==5):
    s = ''

  elif(key==6):
    l = getParameter(0)
    s = l

  else:
    l = getParameter(0)
    s = l

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
  if(key==0):
    l = getParameter(0)
    s = l

  elif(key==1):
    l = getParameter(0)
    r = getParameter(2)
    e = getParameter(3)
    s = l + ',' + r + ',' + e

  elif(key==2):
    l = getParameter(0)
    initial = getParameter(4)
    s = l + ',' + initial

  elif(key==3):
    l = getParameter(0)
    gsa = getParameter(5)
    l1 = getParameter(6)
    l2 = getParameter(7)
    s = l + ',' + gsa + ',' + l1 + ',' + l2

  elif(key==4):
    l = getParameter(0)
    m = getParameter(1)
    s = l + ',' + m

  elif(key==5):
    l = getParameter(0)
    b1 = getParameter(8)
    b2 = getParameter(9)
    e = getParameter(3)
    s = l + ',' + b1 + ',' + b2 + ',' + e

  elif(key==6):
    l = getParameter(0)
    lr = getParameter(10)
    a = getParameter(4)
    l1 = getParameter(6)
    l2 = getParameter(7)
    s = l + ',' + lr + ',' + a + ',' + l1 + ',' + l2

  else:
    l = getParameter(0)
    d = getParameter(11)
    m = getParameter(1)
    e = getParameter(3)
    s = l + ',' + d + ',' + m + ',' + e

  return s

