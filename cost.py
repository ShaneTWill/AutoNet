#!/usr/bin/env python

import sys
import os


# Dictionary of possible cost functions.
__metrics = {
            0 : 'RMSE: Root Mean Squared Error'
            ,1 : 'MSE: Mean Squared Error'
            ,2 : 'MAE: Mean Absolute Error'
            ,3 : 'MAPE: Mean Absolute Percent Error'
            }

# Dictionary of conditional boolean statements used.
__conditions = {
                0: (lambda i: isinstance(i, int) and (-1 < int(i) < len(__metrics)))
                ,1: (lambda i: isinstance(i, int) and int(i) > 0)
                }

# Dictionary of error message strings.
__errors = {
            0: 'Sorry that was not an integer, or an option, please try again.'
            ,1: 'Sorry was not an integer.'
            }

# Dictionary of prompt message strings.
__prompts = {
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

    arry = []
    arry.append('[')
    arry = getData(0,ary)
    arry = getData(1,ary)
    arry.append(']')

    return ''.join(arry)


def getData(code,ary):
    """
    Uses data that the user inputs and and returns the modified
    string ary. For the creation of a list of values.

    Keyword arguments:
    code -- An integer that is used to pull the correct data from the dictionaries
            and perform flow control.
    ary -- A list of strings to append the new values to.

    Returns:
    ary -- The inputted string with the data concatenated onto it.
    """

    prompt = __prompts[code] # Prompt string
    condition = __conditions[code] # Conditional Lambda
    error = __errors[code] # Error Message

    print(prompt)
    if code == 0:
        for key, value in __metrics.items():
            print('{}: {}'.format(key,value))

    while True:
        inp = sys.stdin.readline().strip()
        if(condition(inp)):
            ary.append(str(inp))
            ary.append(',')
            break
        else:
            print(error)
        os.system('cls' if os.name == 'nt' else 'clear')

    return ary

