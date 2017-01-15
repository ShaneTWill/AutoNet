#!/bin/python/


'''
  This is a Neural Network created to perform regression.

  Depth: 2
  Inputs: 12
  Outputs: 1
  Date: 15/01/2017
  Author: Shane Will

'''


import tensorflow as tf
import numpy as np
import pandas as pd
import math


#==================================================
# Reserve memory for Inputs and outputs.
#==================================================

X = tf.placeholder('float64',[None,12])
Y = tf.placeholder('float64',[None,1])


#==================================================
# Neural Network
#==================================================

def infer(x):

  with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([12,30], stddev = 1.0/math.sqrt(float(12))), name='weights')
    biases = tf.Variable(tf.zeros([30])+0.1, name='biases')
    h1 = tf.nn.relu(tf.matmul(x,tf.cast(weights,'float64') + tf.cast(biases,'float64')))

  with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([30,15], stddev = 1.0/math.sqrt(float(30))), name='weights')
    biases = tf.Variable(tf.zeros([15])+0.1, name='biases')
    h2 = tf.nn.relu(tf.matmul(h1,tf.cast(weights,'float64') + tf.cast(biases,'float64')))

  with tf.name_scope('hidden3'):
    weights = tf.Variable(tf.truncated_normal([15,10], stddev = 1.0/math.sqrt(float(15))), name='weights')
    biases = tf.Variable(tf.zeros([10])+0.1, name='biases')
    h3 = tf.nn.relu(tf.matmul(h2,tf.cast(weights,'float64') + tf.cast(biases,'float64')))

  with tf.name_scope('output'):
    weights = tf.Variable(tf.truncated_normal([10,1], stddev = 1.0/math.sqrt(float(10))), name='weights')
    biases = tf.Variable(tf.zeros([1]), name='biases')
    out = tf.matmul(h3,tf.cast(weights,'float64') + tf.cast(biases,'float64'))

  return out


#==================================================
# Cost Function
#==================================================

def Cost(pred,act):

  c = tf.reduce_mean(tf.square(tf.sub(pred,act)))

  return c


#==================================================
# Optimizer Function
#==================================================

def Optimize(loss):

  optimizer = tf.train.AdamOptimizer()
  opt = optimizer.minimize(loss)

  return opt


def train(x_, y_):

  pred = infer(X)
  cost = Cost(pred,Y)
  opt = Optimize(cost)
  init = tf.global_variables_initializer()
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    sess.run(init)
    for epoch in xrange(int(500)):
      _,c = sess.run([opt,cost],feed_dict={X:x_,Y:y_})
      if(epoch % 100 ==0):
        saver.save(sess,'infer.ckpt',global_step=epoch)
        print('Epoch: '+str(epoch)+' Cost: '+str(c))



