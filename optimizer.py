#!/usr/bin/env python

import sys
import os

# Dictionary of possible optimization methods.
OPTIMIZERS = {
            0 : 'GradientDescent'
            ,1 : 'Adadelta'
            ,2 : 'Adagrad'
            ,3 : 'AdagradDA'
            ,4 : 'Momentum'
            ,5 : 'Adam'
            ,6 : 'Ftrl'
            ,7 : 'RMSProp'
            }

# Dictionary of parameters and the associated keys to the __example, __invalid,
# and  __conditions dictionaries
PARAMETERS = {
            0: ('LearningRate',0)
            ,1: ('Momentum',1)
            ,2: ('Rho',0)
            ,3: ('Epsilon',0)
            ,4: ('InitialAccumulatorValue',0)
            ,5: ('InitialGradientSquaredAccumulatorValue',0)
            ,6: ('L1RegularizationStrength',1)
            ,7: ('L2RegularizationStrength',1)
            ,8: ('Beta1',1)
            ,9: ('Beta2',1)
            ,10: ('LearningRatePower',2)
            ,11: ('Decay',1)
            }


# Dictionary of examples for the user to know what data to enter.
EXAMPLES = {
        0: 'a float greater than 0.0 ex. 0.05.'
        ,1: 'a float and must be greater than or equal to 0. i.e. 0.01.'
        ,2: 'a float and must be less than or equal to 0. i.e. -0.5.'
        }

# Dictionary of messages to give the user on invalid data entry.
INVALIDS = {
            0: 'not greater than zero'
            ,1: 'less than zero'
            ,2: 'greater than zero'
            }

# Dictionary of lambda expressions for the conditional expressions.
CONDITIONS = {
            0: (lambda i: i.replace('.','',1).isdigit() and (float(i) > 0))
            ,1: (lambda i: i.replace('.','',1).isdigit() and (float(i) >= 0))
            ,2: (lambda i: i.replace('.','',1).isdigit() and (float(i) <= 0))
            }

# Sets used for flow control when you choose to use the default parameters.
DEFAULTS = {
            0: {0,2,3,6,7}
            ,1: {1,5}
            }

PARAMETER_DICTIONARY = {
    0:[0],
    1:[0,2,3],
    2:[0,4],
    3:[0,5,6,7],
    4:[0,1],
    5:[0,8,9,3],
    6:[0,10,4,6,7]
}

OPTIMIZER_STRING_COLLECTIONS = {
    0: [],
    1: [('rho',3),
        ('epsilon',4)
       ],
    2: [('intial_accumulator_value',3)
       ],
    3: [('initial_gradient_squared_accumulator_value',3),
        ('l1_regularization_strength',4),
        ('l2_regularization_strength',5)
       ],
    4: [('momentum',3)
       ],
    5: [('beta1',3),
        ('beta2',4),
        ('epsilon',5)
       ],
    6: [('learning_rate_power',3),
        ('initial_accumulator_value',4),
        ('l1_regularization_strength',5),
        ('l2_regularization_strength',6)
       ],
}


def getParameter(key):
  """
  Asks the user to select the value for a parameter.

  Keyword arguments:
  key -- An integer that corresponds to a key in the param dictonary.

  Returns:
  val -- The value selected by the user.
  """

  val = ''
  keep_running = True
  parameter, constants_key = PARAMETERS.get(key)

  constants = [EXAMPLES,CONDITIONS,INVALIDS]
  examples, condition, invaild = [constant.get(constants_key) for constant in constants]

  print('What value would you like for the {}?'.format(parameter))
  print('Please note that this is {}'.format(example))
  while keep_running:
    inp = sys.stdin.readline().strip()
    if(condition(inp)):
      val = str(inp)
      keep_running = False
    else:
      print('Sorry that was either  not a float or was {}, please try again'.format(invalid))

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

  data_array = list(param[1])
  key = int(data_array[0])
  optimizer = []
  optimizer.append('optimizer = tf.train.{0}Optimizer('.format(OPTIMIZERS.get(key)))
  if(key in DEFAULT.get(0)):
    optimizer.append(')')
  elif(key in DEFAULT.get(1)):
    optimizer.append('learning_rate={0})'.format(data_array[2]))
  else:
    optimizer.append('learning_rate={0}, momentum={1})'.format(data_array[2],data_array[3]))

  return ''.join(optimizer)


def OptimizerString(param):
  """
  Creates the code for a non-default optimizer string and returns it.

  Keyword arguments:
  param -- a python list that contains the optimizer key and parameter settings.

  Returns:
    optimizer -- The optimizer code as a string.
  """
  data_array = list(param[1])
  key = int(data_array[0])
  string_meta_data = None

  optimizer = []
  optimizer.append('optimizer = tf.train.{0}'.format(OPTIMIZERS.get(key)))
  optimizer.append('Optimizer(learning_rate={0}'.format(data_array[2]))

  # Get the string metadata if the key is invalid return the list of
  # [('decay',3),('momentum',4),('epsilon',5)]
  string_meta_data = OPTIMIZER_STRING_COLLECTIONS.get(key,[('decay',3),('momentum',4),('epsilon',5)])

  for string, pos in string_meta_data:
    optimizer.append(', {0}={1}'.format(string,data_array[pos]))

  optimizer.append(')\n')

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

  array = '['
  cns = ''
  df = 0
  key = 0

  array , key = ChooseOptimizer(array)
  array , df = SetDefault(array)

  if(df == 1):
    array += getOptimizerDefaultParameters(key)
  else:
    array += getOptimizerParameters(key)

  array += ']'
  os.system('cls' if os.name == 'nt' else 'clear')
  return ary


def ChooseOptimizer(array):
  """
  Asks the user to choose an optimizer from the dictionary of choices.

  Keyword arguments:
  ary --  A string used to create the list of the optimizer parameters.

  Returns:
    ary -- The modified string ary.
    key -- The chosen optimizer key as an int.
  """

  keep_running = True

  print('What Optimizer from the list below do you want to optimize with?')
  for key,value in OPTIMIZERS.items():
    print('{0}: {1}'.format(key,value))
  while keep_running:
    i = sys.stdin.readline().strip()
    if(i.isdigit() and (-1<int(i)<8)):
      key = int(i)
      array += str(i) + ','
      keep_running = False
    else:
      print('Sorry that was not an integer, or an option, please try again.')

  os.system('cls' if os.name == 'nt' else 'clear')
  return array,key


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
  keep_running = True

  print('Would you like to use the default settings or not?\nDefault: 1\tNot: 0.')
  while keep_running:
    i = sys.stdin.readline().strip()
    if(i.isdigit() and (-1 < int(i) < 2)):
      df = int(i)
      ary = ary + str(i) + ','
      keep_running = False
    else:
      print('Sorry that was not an integer, or an option, please try again.')

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
  if(key in DEFAULTS[1]):
    s = ''
  elif(key in DEFAULTS[0]):
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
    
  data = PARAMETER_DICTIONARY.get(key,[0,11,1,3])
  parameters = [getParameter(val) for val in data]
    
  return ','.join(parameters)


