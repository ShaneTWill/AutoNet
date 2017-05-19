#!/usr/bin/env python

import time
import sys
import os
import cost as cs
import optimizer as op
import model as ml


# Line to specify that the file is a script for python.
header = '#!/usr/bin/env python'

# import statements that occur at the top of the file.
im = 'import tensorflow as tf\nimport math'



def CreateDocString(struct,inp,out,t):
  """
  Creates the document string for the model.

  Keyword arguments:
  struct -- The parameter list for the model.
  inp -- Input parameter list.
  out -- Output parameter list.
  t -- Model type.

  Returns:
  doc -- Document string for the model file.
  """

  doc = '\'\'\'\n  This is a Neural Network created to perform'
  if(t=='R'):
    doc = doc +' regression.'
      
  else:
    doc = doc +' classification.'

  if(len(struct)-1 == -1):
    doc = doc + '\n\n  Depth: 0' 
  
  else:
    doc = doc + '\n\n  Depth: ' + str(len(struct)-1)
  
  doc = doc + '\n  Inputs: ' + str(inp[0][1]) 
  doc = doc + '\n  Outputs: ' + str(out[0][1])
  doc = doc + '\n  Date: ' + str(time.strftime('%d/%m/%Y'))+'\n\n\'\'\''
    
  return doc


def CreateLayer(prvnum,nodes,laynum,a,nlay,m):
  """
  Creates the document string for the model.

  Keyword arguments:
  prvnum -- Number of nodes in the previous layer.
  nodes -- Number of nodes in this layer.
  laynum -- Layer number.
  a -- Activation function key.
  nlay -- Number of layers in the model
  m -- Model type.

  Returns:
  l -- The code for a layer in a neural network as a string.
  """

  if(laynum == (nlay+1)):
    l = '  with tf.name_scope(\'output\'):\n    '
    l = l + 'weights = tf.Variable(tf.truncated_normal(['+str(prvnum)
    l = l + ','+str(nodes)+'], '
    l = l + 'stddev = 1.0/math.sqrt(float('+str(prvnum)
    l = l + '))), name=\'weights\')\n    '
    l = l + 'biases = tf.Variable(tf.zeros(['+str(nodes)
    l = l + ']), name=\'biases\')\n  '

  else:
    l = '  with tf.name_scope(\'hidden'+str(laynum)+'\'):\n    '
    l = l + 'weights = tf.Variable(tf.truncated_normal(['+str(prvnum)
    l = l + ','+str(nodes)+'], '
    l = l + 'stddev = 1.0/math.sqrt(float('+str(prvnum)+'))), name=\'weights\')\n    '
      
    if(a==0):
      l = l + 'biases = tf.Variable(tf.zeros(['+str(nodes)+'])+0.1, name=\'biases\')\n  '
        
    else:
      l = l + 'biases = tf.Variable(tf.zeros(['+str(nodes)+']), name=\'biases\')\n  '
      

  if(int(laynum) == 1 and laynum != (nlay+1)):
    l = l + '  h'+str(laynum)+' = tf.nn.'+ml.act[a]
    l = l + '(tf.matmul(x,tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n\n'

  elif(laynum == (nlay+1) and (laynum-1) == 0 ):
    if(m =='R'):
      l = l + '  out = tf.matmul(x'
      l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\'))\n'

    else:
      l = l + '  out = tf.nn.sigmoid(tf.matmul(x'
      l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'

  elif(laynum == (nlay+1) and (laynum-1) != 0 ):
    if(m =='R'):
      l = l + '  out = tf.matmul(h'+str(laynum-1)
      l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\'))\n'

    else:
      l = l + '  out = tf.nn.sigmoid(tf.matmul(h'+str(laynum-1)
      l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'
      
  else:
    l = l + '  h'+str(laynum)+' = tf.nn.'+ml.act[a]
    l = l + '(tf.matmul(h'+str(laynum-1)
    l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n\n'
    
  return l  


def CreatePlaceholders(inp,out):
  """
  Creates the placeholders for the data.

  Keyword arguments:
  inp -- List of input data and data type.
  out -- List of output data and data type.
  
  Returns:
  x -- The Input placeholder.
  y -- The Output placeholder.
  """

  x = 'X = tf.placeholder('+str(inp[0][0])+',[None,'+str(inp[0][1])+'])\n'
  y = 'Y = tf.placeholder('+str(out[0][0])+',[None,'+str(out[0][1])+'])\n'
    
  return x,y


def CreateNetwork(struct,mdlname,typ,inp,ou):
  """
  Creates the model.

  Keyword arguments:
  struct -- The data parameters for all the data.
  mdlname -- Model name.
  typ -- Model type.
  inp -- List of input parameters.
  ou -- List of output parameters.
  m -- Model type.

  Returns:
  net -- The code for the model as a string.
  """

  net = 'def '+str(mdlname)+'(x):\n\n'
  lay = len(struct)+1  
  for i in range(len(struct)+1):
    if(i ==len(struct) and len(struct)!=0):
      net = net + CreateLayer(prvnum=struct[i-1][0],
                              nodes=ou[0][1],
                              laynum=(i+1),
                              a=struct[i-1][1],
                              nlay=len(struct),
                              m=typ)

    elif(i==0 and len(struct)!=0 ):
      net = net + CreateLayer(prvnum=inp[0][1],
                              nodes=struct[i][0],
                              laynum=(i+1),
                              a=struct[i][1],
                              nlay=len(struct),
                              m=typ)

    elif(i==0 and len(struct)==0):
      net = net + CreateLayer(prvnum=inp[0][1],
                              nodes=ou[0][1],
                              laynum=(i+1),
                              a='',
                              nlay=len(struct),
                              m=typ)

    else:
      net = net + CreateLayer(prvnum=struct[i-1][0],
                              nodes=struct[i][0],
                              laynum=(i+1),
                              a=struct[i][1],
                              nlay=len(struct),
                              m=typ)


  net = net + '\n  return out\n\n\n'
    
  return net


def CreateCost(lrn):
  """
  Creates the cost function for the model

  Keyword arguments:
  lrn -- Python list with containing the cost function, number of epochs, and the optimizer parameters..

  Returns:
  c -- Code for the cost function as a string.
  """

  c = 'def Cost(pred,act):\n\n  '
  if(lrn[0][0]==0):
    c = c + 'c = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred,act))))\n\n  '

  elif(lrn[0][0]==1):
    c = c + 'c = tf.reduce_mean(tf.square(tf.sub(pred,act)))\n\n  ' 

  elif(lrn[0][0]==2):
    c = c + 'c = tf.reduce_mean(tf.abs(tf.sub(pred,act)))\n\n  '

  elif(lrn[0][0]==3):
    c = c + 'c = tf.reduce_mean(tf.abs(tf.div(tf.sub(pred,act),act)))* 100.0\n\n  '

  else:
    c = c + 'c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,act))\n\n  '

  c = c + 'return c\n\n\n'
    
  return c


def CreateSession(epchnum,mdlname):
  """
  Creates the session code for training the model.

  Keyword arguments:
  epchnum -- Number of epochs to train over.
  mdlname -- Name of the model.

  Returns:
  sess -- The code for the training session.
  """

  sess = 'with tf.Session() as sess:\n    '
  sess = sess + 'sess.run(init)\n    '
  sess = sess + 'for epoch in range(int('+str(epchnum[0][1])+')):\n      '
  sess = sess + '_,c = sess.run([opt,cost],feed_dict={X:x_,Y:y_})\n      '
  sess = sess + 'if(epoch % 100 ==0):\n        '
  sess = sess + 'saver.save(sess,\''+mdlname+'.ckpt\',global_step=epoch)\n        '
  sess = sess + 'print(\'Epoch: \'+str(epoch)+\' Cost: \'+str(c))\n'
    
  return sess


def CreateOptimizer(lrn):
  """
  Creates the optimizer code for the model.

  Keyword arguments:
  lrn -- List of optimizer parameters.

  Returns:
  o -- The code for the neural network optimizer.
  """

  o = 'def Optimize(loss):\n\n' 
  
  if(lrn[1][1] == 1):
    o = o + op.OptimizerDefaultString(lrn) + '  '
      
  else:
    o = o + op.OptimizerString(lrn) + '  '
            
  o = o + 'opt = optimizer.minimize(loss)\n\n  '
  o = o + 'return opt'
    
  return o


def CreateModel(x,y,ty,hidden,learn,filename,modelname):
  """
  Creates the file that holds the model, optimizer, cost function, and train function using the 
  specifications provided.

  Keyword arguments:
  x -- Input data parameter list.
  y -- Output data parameter list.
  ty -- Model type.
  hidden -- List of hidden layer parameters.
  learn -- The optimization parameter list(Cost, number of epochs, and optimizer parameter).
  filename -- Name of the file.
  modelname -- Name of the model
  """
  with open(str(filename)+'.py','w') as f:
      f.write(header)
      [f.write('\n') for _ in range(3)]
      f.write(CreateDocString(struct=hidden,inp=x,out=y,t=ty))
      [f.write('\n') for _ in range(3)]
      f.write(im)
      [f.write('\n') for _ in range(3)]
      f.write('#')
      [f.write('=') for _ in range(50)]
      f.write('\n')
      f.write('# Reserve memory for Inputs and outputs.\n#')
      [f.write('=') for _ in range(50)]
      f.write('\n\n')
      i,o = CreatePlaceholders(inp=x,out=y)
      f.write(i)
      f.write(o)
      [f.write('\n') for _ in range(2)]
      f.write('#')
      [f.write('=') for _ in range(50)]
      f.write('\n')
      f.write('# Neural Network\n#')
      [f.write('=') for _ in range(50)]
      f.write('\n\n')
      f.write(CreateNetwork(struct=hidden,mdlname=modelname,typ=ty,inp=x,ou=y))
      f.write('#')
      [f.write('=') for _ in range(50)]
      f.write('\n')
      f.write('# Cost Function\n#')
      [f.write('=') for _ in range(50)]
      f.write('\n\n')
      f.write(CreateCost(lrn=learn))
      f.write('#')
      [f.write('=') for _ in range(50)]
      f.write('\n')
      f.write('# Optimizer Function\n#')
      [f.write('=') for _ in range(50)]
      f.write('\n\n')
      f.write(CreateOptimizer(learn))
      [f.write('\n') for _ in range(3)]
      f.write(CreateTrainer(mdl=modelname,lrn=learn))
      [f.write('\n') for _ in range(3)]


def CreateTrainer(mdl,lrn):
  """
  Creates the train function.

  Keyword arguments:
  lrn -- The optimizer parameter list.
  mdl -- Name of the model

  Returns:
  s -- The train function as a string.
  """

  s = 'def train(x_, y_):\n\n  '
  s = s + 'pred = '+mdl+'(X)\n  '
  s = s + 'cost = Cost(pred,Y)\n  '
  s = s + 'opt = Optimize(cost)\n  '
  s = s + 'init = tf.global_variables_initializer()\n  '
  s = s + 'saver = tf.train.Saver(tf.global_variables())\n\n  ' 
  s = s + CreateSession(epchnum=lrn,mdlname=mdl)
      
  return s


def run():
  """
  Walks the user through creating the lists of data needed to make a neural network and then writes the model file out.
  """

  os.system('cls' if os.name == 'nt' else 'clear')
  print('This program will create a nerual network of your design.')
  print('Just follow the instructions and give valid inputs.')
  print('If you do not wish to continue enter simply type -1 then press the Enter key.')
  i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
  if(i=='-1'):
    sys.exit()
  inp,out,m = ml.getModelBasics()
  hid = ml.getHiddenLayers()
  lrn = getTrainingSettings()
  print('What should the file be named?')
  print('Please do not inlcude the file extension i.e. do not include the \'.py\'')
  fl = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
  print('Name the model.')
  mdl = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
  print('Creating Model...')
  CreateModel(x=inp,y=out,ty=m,hidden=hid,learn=lrn,filename=fl,modelname=mdl)


def getTrainingSettings():
  """
  Walks the user through selecting a cost function and an optimization algorithm.  

  Returns:
  A python array that is populated with the cost function parameters and the optimizer parameters.
  """

  ary ='['
  e =']'
  met = cs.getMetric()
  train = op.getOptimizer()
  
  ary = ary + met +','+ train + e

  return eval(ary)
  

