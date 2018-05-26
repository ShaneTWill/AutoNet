#!/usr/bin/env python

import sys
import os


# Dictionary of possible cost functions.
METRICS = {
            0 : 'RMSE: Root Mean Squared Error'
            ,1 : 'MSE: Mean Squared Error'
            ,2 : 'MAE: Mean Absolute Error'
            ,3 : 'MAPE: Mean Absolute Percent Error'
            }

# Dictionary of conditional boolean statements used.
CONDITIONS = {
            0: (lambda i: i.isdigit() and (-1 < int(i) < len(_metrics)))
            ,1: (lambda i: i.isdigit() and int(i) > 0)
            }

# Dictionary of error message strings.
ERRORS = {
        0: 'Sorry that was not an integer, or an option, please try again.'
        ,1: 'Sorry was not an integer.'
        }

# Dictionary of prompt message strings.
PROMPTS = {
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

    array = []
    array.append('[')
    for num in range(2):
        array = getData(num,array)
    array.append(']')

    return ''.join(array)


def getData(key,input_array):
    """
    Uses data that the user inputs and and returns the modified
    string ary. For the creation of a list of values.

    Keyword arguments:
    key -- An integer that is used to pull the correct data from the dictionaries
            and perform flow control.

    input_array -- A list of strings to append the new values to.

    Returns:
    array -- The inputted string with the data concatenated onto it.
    """

    keep_running = True
    constants = [PROMPTS,CONDITIONS,ERRORS]

    array = list(input_array)
    prompt, condition, error = [constant.get(key) for constant in constants]

    print(prompt)
    if key == 0:
        for k, v in METRICS.items():
            print('{}: {}'.format(k,v))

    while keep_running:
        user_input = sys.stdin.readline().strip()
        if(condition(user_input)):
            for item in [str(user_input),',']:
                array.append(item)
            keep_running = False
        else:
            print(error)

    os.system('cls' if os.name == 'nt' else 'clear')

    return array

