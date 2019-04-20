#!/usr/bin/env python

import time
import sys
import os
import cost as cs
import optimizer as op
import model as ml


# Line to specify that the file is a python script.
HEADER = '#!/usr/bin/env python'

# import statements that occur at the top of the file.
IMPORTS = 'import tensorflow as tf\nimport math'

# Dictionary of possible cost functions
COST = {
        0 : 'tf.sqrt(tf.reduce_mean(tf.square(tf.sub(pred,act))))'
        ,1 : 'tf.reduce_mean(tf.square(tf.sub(pred,act)))'
        ,2 : 'tf.reduce_mean(tf.abs(tf.sub(pred,act)))'
        ,3 : 'tf.reduce_mean(tf.abs(tf.div(tf.sub(pred,act),act))) * 100.0'
        ,4 : 'tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,act))'
        }

BIASES = 'biases = tf.Variable(tf.zeros([{0}]){1}, name=\'biases\')'

OUTPUT = 'out = tf.{}({},tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\'))'

HIDDEN_LAYER = 'h{} = tf.nn.{}(tf.matmul({},tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))'

MEMORY_LOCATION = '{} = tf.placeholder({},[None,{}])\n'

WEIGHTS = 'weights = tf.Variable(tf.truncated_normal([{0},{1}], stddev = 1.0/math.sqrt(float({0}))), name=\'weights\')'

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
  doc = '''\"\"\"\n  This is a Neural Network created to perform {}.\n\n  Depth: {}\n  Inputs: {}\n  Outputs: {}\n  Date: {}\n\"\"\"'''

  if(t is 'R'):
    modeltype = 'regression'

  if(len(struct)-1 is -1):
    depth = 0

  return doc.format(modeltype,depth,inputs,outputs,date)


def CreateLayer(nodesInPreviousLayer, nodesInLayer, currentLayer, activationKey, numberOfLayers, modelType):
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

  previousLayer = currentLayer - 1
  last_layer = currentLayer is numberOfLayers + 1
  only_layer = previousLayer is 0

  code = '''\n  with tf.name_scope(\'{}\'):\n    {}\n    {}\n    {}\n'''

  activation = ml.ACTIVATIONS.get(activationKey)
  weights = __CreateWeights(nodesInPreviousLayer,nodesInLayer)
  biases = __CreateBiases(activationKey,nodesInLayer)

  if last_layer:
    layer = 'linearModel' if only_layer else 'output'
    equation = __CreateOutputEquation(modelType,previousLayer,currentLayer,numberOfLayers)
  else:
    layer = 'hidden{}'.format(currentLayer)
    equation = __CreateHiddenLayerEquation(currentLayer,numberOfLayers,activation)

  return code.format(layer,weights,biases,equation)

def __CreateWeights(nodesInPreviousLayer,nodesInLayer):
  return WEIGHTS.format(nodesInPreviousLayer,nodesInLayer)

def __CreateOutputEquation(modelType,prevLayNum,currentLayNum,numberOfLayers):
  is_last_layer = currentLayNum is numberOfLayers + 1
  only_layer = prevLayNum is 0
  regression = is_last_layer and only_layer

  function = 'nn.sigmoid' if modelType is 'C' else 'matmul'
  input_values = 'x' if regression else 'h{}'.format(prevLayNum)

  return OUTPUT.format(function,input_values)


def __CreateHiddenLayerEquation(layerNumber,numberOfLayers,activation):
  first_layer = int(layerNumber) is 1
  more_than_one_layer = layerNumber is not numberOfLayers + 1

  inputs = 'x' if first_layer and more_than_one_layer else 'h{}'.format(layerNumber-1)

  return HIDDEN_LAYER.format(layerNumber,activation,inputs)


def __CreateBiases(activationKey,layerNodeCount):
  jitter = ' + 0.1' if activationKey is 0 else ''

  return BIASES.format(layerNodeCount,jitter)


def CreatePlaceholder(data,control):
  """
  Creates a placeholder for the data.

  Keyword arguments:
  data -- List of input data and data type.
  control -- A binary flag to control which placeholder to return.

  Returns:
  placeholder -- The Output placeholder.
  """

  variable = 'Y' if control is 1 else 'X'
  data_type, size = list(data[0])

  return MEMORY_LOCATION.format(variable,data_type,size)


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
  length_of_struct = len(struct)
  numberOfLayers = length_of_struct + 1
  network.append('def {0}(x):\n'.format(mdlname))
  for i in range(numberOfLayers):
    currentLayer = (i+1)
    if(i is 0):
      inputs = inp[0][1]
      outputs, param = (struct[i][0],struct[i][1]) if length_of_struct is not 0 else (ou[0][1],0)
    else:
      inputs = struct[i-1][0]
      outputs, param = (ou[0][1],struct[i-1][1]) if length_of_struct is i  else (struct[i][0],struct[i][1])

    parameters = [inputs,outputs,currentLayer,param,length_of_struct,typ]

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

  return cost'''

  key = lrn[0][0] # Dictionary key
  equation = COST.get(key) # Cost function
  return cost.format(equation)


def CreateSession(epchnum,mdlname):
  """
  Creates the session code for training the model.

  Keyword arguments:
  epchnum -- Number of epochs to train over.
  mdlname -- Name of the model.

  Returns:
  session -- The code for the training session.
  """

  epochs = epchnum[0][1] # number of total epochs
  session = '''with tf.Session() as sess:
  sess.run(init)
  for epoch in range({0}):
    _,c = sess.run([opt,cost], feed_dict={{X:x_,Y:y_}})
    if (epoch % 100 == 0):
      saver.save(sess,\'{1}.ckpt\',global_step=epoch)
      print(\'Epoch: {{0}}\\tCost: {{1}}\'.format(epoch,c))'''

  return session.format(epochs,mdlname)


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

  return opt'''

  default = int(lrn[1][1])

  data = op.OptimizerString(lrn)
  if(default == 1):
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
    f.write(HEADER)
    [f.write('\n') for _ in range(3)]
    f.write(CreateDocString(struct=hidden,inp=x,out=y,t=ty))
    [f.write('\n') for _ in range(3)]
    f.write(IMPORTS)
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
    f.write('\n\n')
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
  if(i == '-1'):
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

  met = cs.getMetric()
  train = op.getOptimizer()

  array = '[' + met +','+ train + ']'

  return eval(array)


if __name__ == "__main__":
    run()
