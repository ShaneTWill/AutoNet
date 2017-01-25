#!/usr/bin/env python

'''

  This class takes inputs and creates the code for a NeuralNetwork model
  the file it creates is a python file that is written out to the current directory.

'''

import time
import sys
import os
import cost as cs
import optimizer as op



# Line to specify that the file is a script for python.
header = '#!/usr/bin/env python'

# import statements that occur at the top of the file.
im = 'import tensorflow as tf\nimport math'


# Dictionary of possible activation functions.
act = {
        0 : 'relu',
        1 : 'relu6',
        2 : 'crelu',
        3 : 'elu',
        4 : 'softplus',
        5 : 'softsign'
        }

# Document string about the model in the file.
def CreateDocString(struct,inp,out,t):
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
  doc = doc + '\n  Date: ' + str(time.strftime('%d/%m/%Y')) 
  doc = doc + '\n  Author: Shane Will\n\n\'\'\''
    
  return doc


# Creates layers for the model.
def CreateLayer(prvnum,nodes,laynum,a,nlay,m):
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
    l = l + '  h'+str(laynum)+' = tf.nn.'+act[a]
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
    l = l + '  h'+str(laynum)+' = tf.nn.'+act[a]
    l = l + '(tf.matmul(h'+str(laynum-1)
    l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n\n'
    
  return l  

  
# Creates the place holders for inputs and outputs.
def CreatePlaceholders(inp,out):
  x = 'X = tf.placeholder('+str(inp[0][0])+',[None,'+str(inp[0][1])+'])\n'
  y = 'Y = tf.placeholder('+str(out[0][0])+',[None,'+str(out[0][1])+'])\n'
    
  return x,y


# Creates the neural network.
def CreateNetwork(struct,mdlname,typ,inp,ou):
  net = 'def '+str(mdlname)+'(x):\n\n'
  lay = len(struct)+1  
  i=0
  while(i < (len(struct))+1):
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

    i+=1
      
  net = net + '\n  return out\n\n\n'
    
  return net


# Creates the cost function.
def CreateCost(lrn):
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


# Writes to code for the actual session running to train of the model.
def CreateSession(epchnum,mdlname):
  sess = 'with tf.Session() as sess:\n    '
  sess = sess + 'sess.run(init)\n    '
  sess = sess + 'for epoch in xrange(int('+str(epchnum[0][1])+')):\n      '
  sess = sess + '_,c = sess.run([opt,cost],feed_dict={X:x_,Y:y_})\n      '
  sess = sess + 'if(epoch % 100 ==0):\n        '
  sess = sess + 'saver.save(sess,\''+mdlname+'.ckpt\',global_step=epoch)\n        '
  sess = sess + 'print(\'Epoch: \'+str(epoch)+\' Cost: \'+str(c))\n'
    
  return sess


# Creates the optimizer for the training of the model.
def CreateOptimizer(lrn):
  o = 'def Optimize(loss):\n\n' 
  
  if(lrn[1][1] == 1):
    o = o + OptimizerDefaultString(lrn) + '  '
      
  else:
    o = o + OptimizerString(lrn) + '  '
            
  o = o + 'opt = optimizer.minimize(loss)\n\n  '
  o = o + 'return opt'
    
  return o


# Creates the optimizer if default option was choosen.
def OptimizerDefaultString(lrn):
  o = '  '
  if(lrn[1][0]==0):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'

  elif(lrn[1][0]==1):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer()\n'
      
  elif(lrn[1][0]==2):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
  elif(lrn[1][0]==3):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
  elif(lrn[1][0]==4):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', momentum='+str(lrn[1][3])+')\n'
      
  elif(lrn[1][0]==5):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer()\n'
      
  elif(lrn[1][0]==6):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
  else:
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
    
  return o


# Creates the optimizer string if default option is not chosen.
def OptimizerString(lrn):
  o = '  '
  if(lrn[1][0]==0):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
  elif(lrn[1][0]==1):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', rho='+str(lrn[1][3])
    o = o + ', epsilon='+str(lrn[1][4])+')\n'
      
  elif(lrn[1][0]==2):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', initial_accumulator_value='+str(lrn[1][3])+')\n'
      
  elif(lrn[1][0]==3):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', initial_gradient_squared_accumulator_value='+str(lrn[1][3])
    o = o + ', l1_regularization_strength='+str(lrn[1][4])
    o = o + ', l2_regularization_strength='+str(lrn[1][5])+')\n'
      
  elif(lrn[1][0]==4):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', momentum='+str(lrn[1][3])+')\n'
      
  elif(lrn[1][0]==5):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', beta1='+str(lrn[1][3])
    o = o + ', beta2='+str(lrn[1][4])
    o = o + ', epsilon='+str(lrn[1][5])+')\n'
      
  elif(lrn[1][0]==6):
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', learning_rate_power='+str(lrn[1][3])
    o = o + ', initial_accumulator_value='+str(lrn[1][4])
    o = o + ', l1_regularization_strength='+str(lrn[1][5])
    o = o + ', l2_regularization_strength='+str(lrn[1][6])+')\n'
      
  else:
    o = o + 'optimizer = tf.train.'+op.opt[lrn[1][0]]
    o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
    o = o + ', decay='+str(lrn[1][3])
    o = o + ', momentum='+str(lrn[1][4])
    o = o + ', epsilon='+str(lrn[1][5])+')\n'
    
    return o

# This method actually writes the file out.
def CreateModel(x,y,ty,hidden,learn,filename,modelname):
  fl = str(filename)+'.py'
  f = open(fl,'w')
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
  f.close()

# Creates the train method.
def CreateTrainer(mdl,lrn):
  s = 'def train(x_, y_):\n\n  '
  s = s + 'pred = '+mdl+'(X)\n  '
  s = s + 'cost = Cost(pred,Y)\n  '
  s = s + 'opt = Optimize(cost)\n  '
  s = s + 'init = tf.global_variables_initializer()\n  '
  s = s + 'saver = tf.train.Saver(tf.global_variables())\n\n  ' 
  s = s + CreateSession(lrn,mdl)
      
  return s

# Holds to code to walk a user through creating the right inputs for file creation.
def run():
  os.system('cls' if os.name == 'nt' else 'clear')
  print('This program will create a nerual network of your design.')
  print('Just follow the instructions and give valid inputs.')
  print('If you do not wish to continue enter simply type -1 then press the Enter key.')
  i = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
  if(i=='-1'):
    sys.exit()
  inp,out,m = getInputs()
  hid = getHiddenLayers(out)
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


# Populates the array of parameters the training optimizer needs.
def getTrainingSettings():
  ary ='['
  e =']'
  met = cs.getMetric()
  train = op.getOptimizer()
  
  ary = ary + met +','+ train + e

  return eval(ary)
  

# Asks the user to choose an activation function for a layer.
def getActivationFunction(cnt,cns):
  n = 0
  print('For layer '+str(cnt+1)+'.\n'+str(act))
  print('That list is the activation functions supported what is this layer made of?')
  while True:
    n = sys.stdin.readline().strip()
    if((n.isdigit()) and (-1 < int(n) < 6)):
      cns = cns + str(n) + ']'
      break
        
    else:
      print('Sorry that was not an integer please try again.')
  os.system('cls' if os.name == 'nt' else 'clear')
    
  return cns


# Asks the user how many nodes are in a layer.
def getNumberofNodes(cnt,cns):
  n = 0
  print('For layer '+str(cnt+1)+'.\nHow many nodes are there?')
  while True:
    n = sys.stdin.readline().strip()
    if(n.isdigit()):
      cns = '['+str(n)+','
      break
        
    else:
      print('Sorry that was either not an integer, or an option, please try again.')
      n = sys.stdin.readline().strip()
  os.system('cls' if os.name == 'nt' else 'clear')
            
  return cns


# Asks the user for the number of layers in the neural network.
def getNumberOfLayers(cnt):
  print('Now how many hidden layers  are there?')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      break
    else:
      print('Sorry that was not an integer.')
  os.system('cls' if os.name == 'nt' else 'clear')

  return i


# Asks the user for the data on the hidden layers.
def getHiddenLayers(out):
  ary = '['
  cnt = 0
  cns = ''
  
  i = getNumberOfLayers(cnt)
  while(cnt < int(i)):
    cns = getNumberofNodes(cnt,cns)
    cns = getActivationFunction(cnt,cns)
    if((cnt+1) != int(int(i))):
      ary += cns + ','
      cns = ''

    else:
      ary += cns + ']'
      
    cnt += 1
  
  if(int(i)==0):
    ary = ary + ']'

  e = eval(ary)
    
  return e


# Asks the user for the model type.
def getModelType():
  t = ''
  m = ''
  print('First thing is first: enter \'R\' for a regression model, and \'C\' for a classification problem.')
  while True:
    i = sys.stdin.readline().strip()
    if(i=='R'):
      t = '\'float64\''
      m = i
      break
    
    elif(i=='C'):
      t = '\'float64\''
      m = i
      break
        
    else:
      print('Sorry that is not an option try again.')

  os.system('cls' if os.name == 'nt' else 'clear')
      
  return t,m


# Asks the user for the number of inputs.
def getInputCount():
  n = ''
  print('Now how many inputs are there?\nPlease enter an integer like 1.')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      n = i
      break
        
    else:
      print('Sorry that is not an option try again.')

  os.system('cls' if os.name == 'nt' else 'clear')
      
  return n

  
# Asks the user for the number of outputs.
def getOutputCount():
  o = ''
  print('How many outputs are there?')
  print('If this is a classification model that uses one-hot encoding enter number of possibile values.')
  while True:
    i = sys.stdin.readline().strip()
    if(i.isdigit()):
      o = i
      break
        
    else:
      print('Sorry that is not an option try again.')

  os.system('cls' if os.name == 'nt' else 'clear')

  return o


# Gets the data on number of inputs, outputs, and model type.
def getInputs():
  t = ''
  n = 0
  o = 0
  m = ''
  t , m = getModelType()
  n = getInputCount()
  o = getOutputCount()
 
  return [[t,n]],[[t,o]],m


#run()
