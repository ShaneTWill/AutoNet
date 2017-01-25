#!/usr/bin/env python

#=========================================================================#
#									  #
# This file holds all the code for choosing the model optimizer settings. #
#									  #
#=========================================================================#

import sys
import os
import fileCreation as an

# Asks the user to choose an optimizer.
def ChooseOptimizer(ary):
  while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print('What Optimizer from the list do you want to optimize by?\n'+str(an.opt))
    print('Please enter the integer corresponding to the correct Optimizer.')
    i = sys.stdin.readline().strip()
    if(i.isdigit() and (-1<int(i)<8)):
      temp = int(i)
      ary = ary + str(i) + ','
      break
  
    else:
      print('Sorry that was not an integer, or an option, please try again.')

  os.system('cls' if os.name == 'nt' else 'clear')

  return ary,temp


# Asks the user if they want to use the default settings or not.
def SetDefault(ary):
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
def getOptimizerDefaultParameters(temp):
  s = ''
  if(temp==0):
    l = getLearningRate()
    s = l

  elif(temp==1):
    s = ''

  elif(temp==2):
    l = getLearningRate()
    s = l

  elif(temp==3):
    l = getLearningRate()
    s = l

  elif(temp==4):
    l = getLearningRate()
    m = getMomentum()
    s = l + ',' + m

  elif(temp==5):
    s = ''

  elif(temp==6):
    l = getLearningRate()
    s = l

  else:
    l = getLearningRate()
    s = l

  return s


# Gets the optimizer values needed if default is not chosen
def getOptimizerParameters(temp):
  s = ''
  if(temp==0):
    l = getLearningRate()
    s = l

  elif(temp==1):
    l = getLearningRate()
    r = getRho()
    e = getEpsilon()
    s = l + ',' + r + ',' + e

  elif(temp==2):
    l = getLearningRate()
    initial = getInitialAccumulatorValue()
    s = l + ',' + initial

  elif(temp==3):
    l = getLearningRate()
    gsa = getInitialGradientSquaredAccumulatorValue()
    l1 = getL1RegularizationStrength()
    l2 = getL2RegularizationStrength()
    s = l + ',' + gsa + ',' + l1 + ',' + l2

  elif(temp==4):
    l = getLearningRate()
    m = getMomentum()
    s = l + ',' + m

  elif(temp==5):
    l = getLearningRate()
    b1 = getBeta1()
    b2 = getBeta2()
    e = getEpsilon()
    s = l + ',' + b1 + ',' + b2 + ',' + e

  elif(temp==6):
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


# Asks the user for the Rho value. 
def getLearningRate():
  val = ''
  print('Now what is the learning rate?\nEnter a float ex. 0.005.')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float)):
      val = str(i)
      break
  
    else:
      print('Sorry that was not an a float, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')

  return val


# Asks the user for the Rho value. 
def getRho():
  val = ''
  print('What value would you like for rho?\nPlease note that this is a float. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float)):
      val = str(i)
      break
        
    else:
      print('Sorry that was not an a float, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
      
  return val


# Asks the user for the Epsilon value.
def getEpsilon():
  val = ''
  print('What value would you like for epsilon?')
  print('Please note that this is a float. i.e. 0.01')
  while True:
    i = sys.stdin.readline().strip()
    if(isinstance(float(i),float)):
      val = str(i)
      break
    
    else:
      print('Sorry that was not an a float, please try again.')
    i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')

  return val


# Asks the user for the Initial Accumulator value.
def getInitialAccumulatorValue():
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


# Asks the user for the Gradient Squared Accumulator value.    
def getInitialGradientSquaredAccumulatorValue():
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


# Asks the user for the L1 Regularization Strength value. 
def getL1RegularizationStrength():
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


# Asks the user for the L2 Regularization Strength value.
def getL2RegularizationStrength():
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


# Asks the user for the Beta1 value.
def getBeta1():
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


# Asks the user for the Momentum value.
def getMomentum():
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

 
# Asks the user for the Learning Rate Power value.
def getLearningRatePower():
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


# Asks the user for the Decay value.
def getDecay():
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
