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


# Dictionary of possible cost functions
__cost = { 
        0 : 'tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred,act))))'
        ,1 : 'tf.reduce_mean(tf.square(tf.sub(pred,act)))'
        ,2 : 'tf.reduce_mean(tf.abs(tf.sub(pred,act)))'
        ,3 : 'tf.reduce_mean(tf.abs(tf.div(tf.sub(pred,act),act)))* 100.0'
        ,4 : 'tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,act))'
        }



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
  
  modeltype = 'classification'
  depth = len(struct)-1
  inputs = inp[0][1]
  outputs = out[0][1]
  date = time.strftime('%m/%d/%Y')
  
  doc = '''\'\'\'\n  This is a Neural Network created to perform {0}.
  
  Depth: {1}
  Inputs: {2}
  Outputs: {3}
  Date: {4}\n\'\'\'
  '''

  if(t=='R'):
    modeltype = 'regression'

  if(len(struct)-1 == -1):
    depth = 0
 
  data = [modeltype,depth,inputs,outputs,date]
    
  return doc.format(*data)


def CreateLayer(nodesInPreviousLayer,nodesInLayer,currentLayer,activationKey,numberOfLayers,modelType):
  """
  Creates the document string for the model.

  Keyword arguments:
  nodesInPreviousLayer -- Number of nodes in the previous layer.
  nodesInLayer -- Number of nodes in this layer.
  currentLayer -- Layer number.
  activationKey -- Activation function key.
  numberOfLayers -- Number of layers in the model
  modeType -- Model type.

  Returns:
  code -- The code for a layer in a neural network as a string.
  """

  layer = 'hidden{0}'.format(currentLayer)
  previousLayer = currentLayer-1
  activation = ml.activations[activationKey]
  biases = 'biases = tf.Variable(tf.zeros([{0}]), name=\'biases\')'.format(nodesInLayer)
  equation = 'h{0} = tf.nn.{1}(tf.matmul(h{2},tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'.format(currentLayer
                                                                                                                       ,activation
                                                                                                                       ,previousLayer
                                                                                                                       )
  code = '''
    with tf.name_scope(\'{0}\'):
      weights = tf.Variable(tf.truncated_normal([{1},{2}], stddev = 1.0/math.sqrt(float({1}))), name=\'weights\')
      {3}
      {4}'''
  
  if(currentLayer == (numberOfLayers+1) and (previousLayer) == 0):
    layer = 'linearModel'
    equation = __CreateOutputEquation(modelType,previousLayer,currentLayer,numberOfLayers)

  elif(currentLayer == numberOfLayers + 1):
    layer = 'output'
    equation = __CreateOutputEquation(modelType,previousLayer,currentLayer,numberOfLayers)


  else:
    biases = __CreateBiases(activationKey,nodesInLayer)
    equation = __CreateHiddenLayerEquation(currentLayer,numberOfLayers,activation)


  data = [layer,nodesInPreviousLayer,nodesInLayer,biases,equation]

  return code.format(*data)


def __CreateOutputEquation(modelType,prevLayNum,currentLayNum,numberOfLayers):
  equation = 'out = tf.matmul(h{0},tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\'))\n'.format(prevLayNum)
  if(modelType == 'C'):
    equation = 'out = tf.nn.sigmoid(tf.matmul(h{0},tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'.format(prevLayNum)
    if (currentLayNum == (numberOfLayers+1) and (prevLayNum == 0)):
      equation = 'out = tf.nn.sigmoid(tf.matmul(x,tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'
  
  elif(currentLayNum == (numberOfLayers+1) and (prevLayNum == 0)):
    equation = 'out = tf.matmul(x,tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\'))\n'

  return equation


def __CreateHiddenLayerEquation(layerNumber,numberOfLayers,activation):
  equation = 'h{0} = tf.nn.{1}(tf.matmul(h{2},tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'.format(layerNumber
                                                                                                                       ,activation
                                                                                                                       ,layerNumber-1
                                                                                                                       )
  if(int(layerNumber) == 1 and layerNumber != (numberOfLayers+1)):
    equation = 'h{0} = tf.nn.{1}(tf.matmul(x,tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'.format(layerNumber
                                                                                                                      ,activation
                                                                                                                      )
  return equation


def __CreateBiases(activationKey,layerNodeCount):
  biases = 'biases = tf.Variable(tf.zeros([{0}]), name=\'biases\')'.format(layerNodeCount)
  if(activationKey == 0):
    biases = 'biases = tf.Variable(tf.zeros([{0}]) + 0.1 , name=\'biases\')'.format(layerNodeCount)

  return biases


def CreatePlaceholder(data,control):
  """
  Creates a placeholder for the data.

  Keyword arguments:
  data -- List of input data and data type.
  control -- A binary flag to control which placeholder to return.

  Returns:
  placeholder -- The Output placeholder.
  """

  placeholder = 'X = tf.placeholder({0},[None,{1}])\n'
  if(control == 1):
    placeholder = 'Y = tf.placeholder({0},[None,{1}])\n'
    
  return placeholder.format(*(data[0]))


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
  network -- The code for the model as a string.
  """

  network = []
  network.append('def {0}(x):\n'.format(mdlname))
  numberOfLayers = len(struct)+1  
  for i in range(numberOfLayers):
    currentLayer = (i+1)
    if(i ==len(struct) and len(struct)!=0):
      parameters = [struct[i-1][0],ou[0][1],currentLayer,struct[i-1][1],len(struct),typ]

    elif(i==0 and len(struct)!=0 ):
      parameters = [inp[0][1],struct[i][0],currentLayer,struct[i][1],len(struct),typ]

    elif(i==0 and len(struct)==0):
      parameters = [inp[0][1],ou[0][1],currentLayer,0,len(struct),typ]

    else:
      parameters = [struct[i-1][0],struct[i][0],currentLayer,struct[i][1],len(struct),typ]
      
    network.append(CreateLayer(*parameters))

  network.append('\n  return out\n\n\n')
    
  return ''.join(network)


def CreateCost(lrn):
  """
  Creates the cost function for the model

  Keyword arguments:
  lrn -- Python list with containing the cost function, number of epochs, and the optimizer parameters..

  Returns:
  c -- Code for the cost function as a string.
  """
  
  cost = '''def Cost(pred,act):

  cost = {0}

  return cost\n\n'''
  equationKey = lrn[0][0]
    
  return cost.format(__cost[equationKey])


def CreateSession(epchnum,mdlname):
  """
  Creates the session code for training the model.

  Keyword arguments:
  epchnum -- Number of epochs to train over.
  mdlname -- Name of the model.

  Returns:
  session -- The code for the training session.
  """

  steps = epchnum[0][1]
  session = '''with tf.Session() as sess:
    sess.run(init)
    for epoch in range({0}):
      _,c = sess.run([opt,cost], feed_dict={{X:x_,Y:y_}})
      if(epoch % 100 == 0)
        saver.save(sess,\'{1}.ckpt\',global_step=epoch)
        print(\'Epoch: {{0}}\\tCost: {{1}}\'.format(epoch,c))
  '''

  return session.format(steps,mdlname)


def CreateOptimizer(lrn):
  """
  Creates the optimizer code for the model.

  Keyword arguments:
  lrn -- List of optimizer parameters.

  Returns:
  optimize -- The code for the neural network optimizer.
  """
  
  optimize = '''def Optimize(loss):

  {0}
  opt = optimizer.minimize(loss)
    
  return opt
  '''

  data = op.OptimizerString(lrn)
  
  if(lrn[1][1] == 1):
    data = op.OptimizerDefaultString(lrn)
    
  return optimize.format(data)


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
      i = CreatePlaceholder(x,0)
      o = CreatePlaceholder(y,1)
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

  train = '''def train(x_,y_):

    pred = {0}(X)
    cost = Cost(pred,Y)
    opt = Optimize(cost)
    init = tf.gobal_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    {1}'''

  session = CreateSession(epchnum=lrn,mdlname=mdl)
      
  return train.format(mdl,session)


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
  

