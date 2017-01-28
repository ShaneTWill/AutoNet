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
    print('What Optimizer from the list do you want to optimize by?\n'+str(opt))
    print('Please enter the integer corresponding to the correct Optimizer.')
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
  while True:
    i = sys.stdin.readline().strip()
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
    l = getLearningRate()
    s = l

  elif(key==1):
    s = ''

  elif(key==2):
    l = getLearningRate()
    s = l

  elif(key==3):
    l = getLearningRate()
    s = l

  elif(key==4):
    l = getLearningRate()
    m = getMomentum()
    s = l + ',' + m

  elif(key==5):
    s = ''

  elif(key==6):
    l = getLearningRate()
    s = l

  else:
    l = getLearningRate()
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
    l = getLearningRate()
    s = l

  elif(key==1):
    l = getLearningRate()
    r = getRho()
    e = getEpsilon()
    s = l + ',' + r + ',' + e

  elif(key==2):
    l = getLearningRate()
    initial = getInitialAccumulatorValue()
    s = l + ',' + initial

  elif(key==3):
    l = getLearningRate()
    gsa = getInitialGradientSquaredAccumulatorValue()
    l1 = getL1RegularizationStrength()
    l2 = getL2RegularizationStrength()
    s = l + ',' + gsa + ',' + l1 + ',' + l2

  elif(key==4):
    l = getLearningRate()
    m = getMomentum()
    s = l + ',' + m

  elif(key==5):
    l = getLearningRate()
    b1 = getBeta1()
    b2 = getBeta2()
    e = getEpsilon()
    s = l + ',' + b1 + ',' + b2 + ',' + e

  elif(key==6):
    l = getLearningRate()
    lr = getLearningRatePower()
    a = getInitialAccumulatorValue()
    l1 = getL1RegularizationStrength()
    l2 = getL2RegularizationStrength()
    s = l + ',' + lr + ',' + a + ',' + l1 + ',' + l2

  else:
    l = getLearningRate()
    d = getDecay()
    m = getMomentum()
    e = getEpsilon()
    s = l + ',' + d + ',' + m + ',' + e

  return s


def getLearningRate():
  """
  Asks the user to select the value for the Learning Rate.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  print('Now what is the learning rate?\nEnter a float ex. 0.005.')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and (float(i) >0)):
      val = str(i)
      break
  
    else:
      print('Sorry that was not an a float, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')

  return val


def getRho():
  """
  Asks the user to select a value for Rho.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  print('What value would you like for rho?\nPlease note that this is a float. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and (float(i) >0)):
      val = str(i)
      break
        
    else:
      print('Sorry that was not an a float, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def getEpsilon():
  """
  Asks the user to set the Epsilon value.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  print('What value would you like for epsilon?')
  print('Please note that this is a float. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and (float(i) >0)):
      val = str(i)
      break
    
    else:
      print('Sorry that was not an a float, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')

  return val


def getInitialAccumulatorValue():
  """
  Asks the user for the Initial Accumulator Value.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  print('What value would you like for intial_accumulator_value?')
  print('Please note that this is a float and must be greater than zero. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) > 0.0):
      val = str(i)
      break
        
    else:
      print('Sorry that was either not an a float, or not greater than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
    
  return val


def getInitialGradientSquaredAccumulatorValue():
  """
  Asks the user to select the value for Initial Gradient Squared Accumulator Value.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  print('What value would you like for intial_gradient_squared_accumulator_value?')
  print('Please note that this is a float and must be greater than 0. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) > 0.0):
      val = str(i)
      break
        
    else:
      print('Sorry that was either not an a float, or not greater than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def getL1RegularizationStrength():
  """
  Asks the user to select a value for the L1 Regularization Strength.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  print('What value would you like for l1_regularization_strength?')
  print('Please note that this is a float and must be greater than or equal to 0. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) >= 0.0):
      val = str(i)
      break
        
    else:
      print('Sorry that was either not an a float, or less than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def getL2RegularizationStrength():
  """
  Asks the user to select the value for L2 Regularization Strength.

  Returns:
  val -- The value selected by the user. 
  """

  val = ''
  print('What value would you like for l2_regularization_strength?')
  print('Please note that this is a float and must be greater than or equal to 0. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) >= 0.0):
      val = str(i)
      break
  
    else:
      print('Sorry that was either not an a float, or less than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def getBeta1():
  """
  Asks the user to select the Beta1 value.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  print('What value would you like for beta1?')
  print('Please note that this is a float and must be greater than or equal to 0. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) >= 0.0):
      val = str(i)
      break
        
    else:
      print('Sorry that was either not an a float, or less than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val
    

# Asks the user for the Beta2 value.
def getBeta2():
  """
  Asks the user to select the Beta2 value.

  Returns:
  val -- The values selected by the user.
  """

  val = ''
  print('What value would you like for beta2?')
  print('Please note that this is a float and must be greater than or equal to 0. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) >= 0.0):
      val = str(i)
      break
  
    else:
      print('Sorry that was either not an a float, or less than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def getMomentum():
  """
  Asks the user for the value to set momentum to.

  Returns:
  val -- The momentum value selected by the user.
  """

  val = ''
  print('What value would you like for momentum?')
  print('Please note that this is a float and must be greater than or equal to 0. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) >= 0.0):
      val = str(i)
      break
        
    else:
      print('Sorry that was either not an a float, or less than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def getLearningRatePower():
  """
  Asks the user for the value for Learning Rate Power.

  Returns:
  val -- the value selected by the user for Learning Rate Power.
  """

  val = ''
  print('What value would you like for learning_rate_power?')
  print('Please note that this is a float and must be less than or equal to 0. i.e. -0.5')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) <= 0.0):
      val = str(i)
      break
        
    else:
      print('Sorry that was either not an a float, or greater than 0, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


def getDecay():
  """
  Asks the user to set the decay value.

  Returns:
  val -- The user selected value for decay.
  """

  val = ''
  print('What value would you like for decay?')
  print('Please note that this is a float and must be less than or equal to 0. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float) and float(i) >= 0.0):
      val = str(i)
      break
  
    else:
      print('Sorry that was either not an a float, or less than 0, please try again.')
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val
