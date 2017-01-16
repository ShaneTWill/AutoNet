#!/usr/bin/env python

'''

  This class takes inputs and creates the code for a NeuralNetwork model
  the file it creates is a python file that is written out to the current directory.

'''

import time
import sys
import os

class AutoNet:

  # Line to specify that the file is a script for python.
  header = '#!/usr/bin/env python'

  # import statements that occur at the top of the file.
  im = 'import tensorflow as tf\nimport math'

  # Dictionary of Possible metrics
  met = {
          0 : 'RMSE',
          1 : 'MSE',
          2 : 'MAE',
          3 : 'MAPE'
        }

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
  @staticmethod
  def CreateDocString(struct,inp,out,t):
    doc = '\'\'\'\n  This is a Neural Network created to perform'
    if(t=='R'):
      doc = doc +' regression.'
      
    else:
      doc = doc +' classification.'

    doc = doc + '\n\n  Depth: ' + str(len(struct)-1) 
    doc = doc + '\n  Inputs: ' + str(inp[0][1]) 
    doc = doc + '\n  Outputs: ' + str(out[0][1])
    doc = doc + '\n  Date: ' + str(time.strftime('%d/%m/%Y')) 
    doc = doc + '\n  Author: Shane Will\n\n\'\'\''
    
    return doc


  # Creates layers for the model.
  @staticmethod
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
      

    if(int(laynum) == 1):
       l = l + '  h'+str(laynum)+' = tf.nn.'+AutoNet.act[a]
       l = l + '(tf.matmul(x,tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n\n'
      
    elif(laynum == (nlay+1)):
      if(m =='R'):
        l = l + '  out = tf.matmul(h'+str(laynum-1)
        l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\'))\n'

      else:
        l = l + '  out = tf.nn.sigmoid(tf.matmul(h'+str(laynum-1)
        l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n'
      
    else:
      l = l + '  h'+str(laynum)+' = tf.nn.'+AutoNet.act[a]
      l = l + '(tf.matmul(h'+str(laynum-1)
      l = l + ',tf.cast(weights,\'float64\') + tf.cast(biases,\'float64\')))\n\n'
    
    return l  

  
  # Creates the place holders for inputs and outputs.
  @staticmethod
  def CreatePlaceholders(inp,out):
    x = 'X = tf.placeholder('+str(inp[0][0])+',[None,'+str(inp[0][1])+'])\n'
    y = 'Y = tf.placeholder('+str(out[0][0])+',[None,'+str(out[0][1])+'])\n'
    
    return x,y


  # Creates the neural network.
  @staticmethod
  def CreateNetwork(struct,mdlname,typ,inp,ou):
    net = 'def '+str(mdlname)+'(x):\n\n'
    lay = len(struct)+1  
    i=0
    while(i < (len(struct))+1):
      if(i ==len(struct)):
        net = net + AutoNet.CreateLayer(prvnum=struct[i-1][0],
                nodes=ou[0][1],
                laynum=(i+1),
                a=struct[i-1][1],
                nlay=len(struct),
                m=typ)

      elif(i==0):
        net = net + AutoNet.CreateLayer(prvnum=inp[0][1],
                nodes=struct[i][0],
                laynum=(i+1),
                a=struct[i][1],
                nlay=len(struct),
                m=typ)

      else:
        net = net + AutoNet.CreateLayer(prvnum=struct[i-1][0],
                nodes=struct[i][0],
                laynum=(i+1),
                a=struct[i][1],
                nlay=len(struct),
                m=typ)

      i+=1
      
    net = net + '\n  return out\n\n\n'
    
    return net


  # Creates the cost function.
  @staticmethod
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


  # Writes to code for the actual session running to train of the model
  @staticmethod
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
  @staticmethod
  def CreateOptimizer(lrn):
    o = 'def Optimize(loss):\n\n' 
    
    if(lrn[1][1] == 1):
      o = o + AutoNet.OptimizerDefaultString(lrn) + '  '
      
    else:
      o = o + AutoNet.OptimizerString(lrn) + '  '
            
    o = o + 'opt = optimizer.minimize(loss)\n\n  '
    o = o + 'return opt'
    
    return o


  # Creates the optimizer if default option was choosen.
  @staticmethod
  def OptimizerDefaultString(lrn):
    o = '  '
    if(lrn[1][0]==0):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'

    elif(lrn[1][0]==1):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer()\n'
      
    elif(lrn[1][0]==2):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
    elif(lrn[1][0]==3):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
    elif(lrn[1][0]==4):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', momentum='+str(lrn[1][3])+')\n'
      
    elif(lrn[1][0]==5):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer()\n'
      
    elif(lrn[1][0]==6):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
    else:
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
    
    return o


  # Creates the optimizer string if default option is not chosen
  @staticmethod
  def OptimizerString(lrn):
    o = '  '
    if(lrn[1][0]==0):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])+')\n'
      
    elif(lrn[1][0]==1):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', rho='+str(lrn[1][3])
      o = o + ', epsilon='+str(lrn[1][4])+')\n'
      
    elif(lrn[1][0]==2):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', initial_accumulator_value='+str(lrn[1][3])+')\n'
      
    elif(lrn[1][0]==3):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', initial_gradient_squared_accumulator_value='+str(lrn[1][3])
      o = o + ', l1_regularization_strength='+str(lrn[1][4])
      o = o + ', l2_regularization_strength='+str(lrn[1][5])+')\n'
      
    elif(lrn[1][0]==4):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', momentum='+str(lrn[1][3])+')\n'
      
    elif(lrn[1][0]==5):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', beta1='+str(lrn[1][3])
      o = o + ', beta2='+str(lrn[1][4])
      o = o + ', epsilon='+str(lrn[1][5])+')\n'
      
    elif(lrn[1][0]==6):
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', learning_rate_power='+str(lrn[1][3])
      o = o + ', initial_accumulator_value='+str(lrn[1][4])
      o = o + ', l1_regularization_strength='+str(lrn[1][5])
      o = o + ', l2_regularization_strength='+str(lrn[1][6])+')\n'
      
    else:
      o = o + 'optimizer = tf.train.'+AutoNet.opt[lrn[1][0]]
      o = o + 'Optimizer(learning_rate='+str(lrn[1][2])
      o = o + ', decay='+str(lrn[1][3])
      o = o + ', momentum='+str(lrn[1][4])
      o = o + ', epsilon='+str(lrn[1][5])+')\n'
    
    return o

  # This method actually writes the file out.
  @staticmethod
  def CreateModel(x,y,ty,hidden,learn,filename,modelname):
    fl = str(filename)+'.py'
    f = open(fl,'w')
    f.write(AutoNet.header)
    [f.write('\n') for _ in range(3)]
    f.write(AutoNet.CreateDocString(struct=hidden,inp=x,out=y,t=ty))
    [f.write('\n') for _ in range(3)]
    f.write(AutoNet.im)
    [f.write('\n') for _ in range(3)]
    f.write('#')
    [f.write('=') for _ in range(50)]
    f.write('\n')
    f.write('# Reserve memory for Inputs and outputs.\n#')
    [f.write('=') for _ in range(50)]
    f.write('\n\n')
    i,o = AutoNet.CreatePlaceholders(inp=x,out=y)
    f.write(i)
    f.write(o)
    [f.write('\n') for _ in range(2)]
    f.write('#')
    [f.write('=') for _ in range(50)]
    f.write('\n')
    f.write('# Neural Network\n#')
    [f.write('=') for _ in range(50)]
    f.write('\n\n')
    f.write(AutoNet.CreateNetwork(struct=hidden,mdlname=modelname,typ=ty,inp=x,ou=y))
    f.write('#')
    [f.write('=') for _ in range(50)]
    f.write('\n')
    f.write('# Cost Function\n#')
    [f.write('=') for _ in range(50)]
    f.write('\n\n')
    f.write(AutoNet.CreateCost(lrn=learn))
    f.write('#')
    [f.write('=') for _ in range(50)]
    f.write('\n')
    f.write('# Optimizer Function\n#')
    [f.write('=') for _ in range(50)]
    f.write('\n\n')
    f.write(AutoNet.CreateOptimizer(learn))
    [f.write('\n') for _ in range(3)]
    f.write(AutoNet.CreateTrainer(mdl=modelname,lrn=learn))
    [f.write('\n') for _ in range(3)]
    f.close()

  # Creates the train method
  @staticmethod
  def CreateTrainer(mdl,lrn):
    s = 'def train(x_, y_):\n\n  '
    s = s + 'pred = '+mdl+'(X)\n  '
    s = s + 'cost = Cost(pred,Y)\n  '
    s = s + 'opt = Optimize(cost)\n  '
    s = s + 'init = tf.global_variables_initializer()\n  '
    s = s + 'saver = tf.train.Saver(tf.global_variables())\n\n  ' 
    s = s + AutoNet.CreateSession(lrn,mdl)
      
    return s

  # Holds to code to walk a user through creating the right inputs for file creation.
  @staticmethod
  def run():
    os.system('cls' if os.name == 'nt' else 'clear')
    print('This program will create a nerual network of your design.')
    print('Just follow the instructions and give valid inputs.')
    print('If you do not wish to continue enter simply type -1 then press the Enter key.')
    i = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')
    if(i=='-1'):
      sys.exit()
    inp,out,m = AutoNet.getInputs()
    hid = AutoNet.getHiddenLayers(out)
    lrn = AutoNet.getTrainingSettings()
    print('What should the file be named?')
    print('Please do not inlcude the file extension i.e. do not include the \'.py\'')
    fl = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')
    print('Name the model.')
    mdl = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')
    print('Creating Model...')
    AutoNet.CreateModel(x=inp,y=out,ty=m,hidden=hid,learn=lrn,filename=fl,modelname=mdl)


  # Populates the array of parameters the training optimizer needs.
  @staticmethod
  def getTrainingSettings():
    ary ='['
    e =']'
    met = AutoNet.getMetric()
    train = AutoNet.getOptimizer()
    
    ary = ary + met +','+ train + e

    return eval(ary)

  # Asks the user for the metric to optimize by.
  @staticmethod
  def ChooseMetric(ary):
    print('What metric from the list do you want to optimize by?\n'+str(AutoNet.met))
    print('Please enter the integer corresponding to the correct metric.')
    while True:
      i = sys.stdin.readline().strip()
      if(i.isdigit() and (-1<int(i)<4)):
        ary = ary + str(i) + ','
        break
        
      else:
        print('Sorry that was not an integer, or an option, please try again.')
      i = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')
      
    return ary

  # Asks the user for the number of iterations to train over.
  @staticmethod
  def getEpoch(ary):
    print('Finally how many epochs should the model be trained over?')
    while True:
      i = sys.stdin.readline().strip()
      if(i.isdigit()):
        ary = ary + str(i)
        break
        
      else:
        print('Sorry was not an integer.')
        i = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')

    return ary


  # Asks for the input of data for cost metric, and the number of epochs to train over.
  @staticmethod
  def getMetric():
    ary = '['
    e = ']'
    
    ary = AutoNet.ChooseMetric(ary)
    ary = AutoNet.getEpoch(ary)
    ary = ary + e

    return ary


  # Asks the user to choose an optimizer.
  @staticmethod
  def ChooseOptimizer(ary):
    while True:
      os.system('cls' if os.name == 'nt' else 'clear')
      print('What Optimizer from the list do you want to optimize by?\n'+str(AutoNet.opt))
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
  @staticmethod
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


  # Gets the user inputs and data for the Optimizer.
  @staticmethod
  def getOptimizer():
    ary = '['
    cns = ''
    df = 0
    temp = 0

    ary , temp = AutoNet.ChooseOptimizer(ary)
    ary , df = AutoNet.SetDefault(ary)

    if(df==1):
      ary = ary + AutoNet.getDefaults(temp)
      
    else:
      ary = ary + AutoNet.getOptimizerParameters(temp)
    
    ary = ary + ']'     
    os.system('cls' if os.name == 'nt' else 'clear')
    
    return ary


  # Gets the optimizer values needed if default is chosen.
  @staticmethod
  def getDefaults(temp):
    s = ''
    if(temp==0):
      l = AutoNet.getLearningRate()
      s = l

    elif(temp==1):
      s = ''

    elif(temp==2):
      l = AutoNet.getLearningRate()
      s = l

    elif(temp==3):
      l = AutoNet.getLearningRate()
      s = l

    elif(temp==4):
      l = AutoNet.getLearningRate()
      m = AutoNet.getMomentum()
      s = l + ',' + m

    elif(temp==5):
      s = ''

    elif(temp==6):
      l = AutoNet.getLearningRate()
      s = l

    else:
      l = AutoNet.getLearningRate()
      s = l

    return s


  # Gets the optimizer values needed if default is not chosen
  @staticmethod
  def getOptimizerParameters(temp):
    s = ''
    if(temp==0):
      l = AutoNet.getLearningRate()
      s = l

    elif(temp==1):
      l = AutoNet.getLearningRate()
      r = AutoNet.getRho()
      e = AutoNet.getEpsilon()
      s = l + ',' + r + ',' + e

    elif(temp==2):
      l = AutoNet.getLearningRate()
      initial = AutoNet.getInitialAccumulatorValue()
      s = l + ',' + initial

    elif(temp==3):
      l = AutoNet.getLearningRate()
      gsa = AutoNet.getInitialGradientSquaredAccumulatorValue()
      l1 = AutoNet.getL1RegularizationStrength()
      l2 = AutoNet.getL2RegularizationStrength()
      s = l + ',' + gsa + ',' + l1 + ',' + l2

    elif(temp==4):
      l = AutoNet.getLearningRate()
      m = AutoNet.getMomentum()
      s = l + ',' + m

    elif(temp==5):
      l = AutoNet.getLearningRate()
      b1 = AutoNet.getBeta1()
      b2 = AutoNet.getBeta2()
      e = AutoNet.getEpsilon()
      s = l + ',' + b1 + ',' + b2 + ',' + e

    elif(temp==6):
      l = AutoNet.getLearningRate()
      lr = AutoNet.getLearningRatePower()
      a = AutoNet.getInitialAccumulatorValue()
      l1 = AutoNet.getL1RegularizationStrength()
      l2 = AutoNet.getL2RegularizationStrength()
      s = l + ',' + lr + ',' + a + ',' + l1 + ',' + l2

    else:
      l = AutoNet.getLearningRate()
      d = AutoNet.getDecay()
      m = AutoNet.getMomentum()
      e = AutoNet.getEpsilon()
      s = l + ',' + d + ',' + m + ',' + e

    return s


  # Asks the user for the Rho value. 
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
  @staticmethod
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
    

  # Asks the user to choose an activation function for a layer.
  @staticmethod
  def getActivationFunction(cnt,cns):
    n = 0
    print('For layer '+str(cnt+1)+'.\n'+str(AutoNet.act))
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
  @staticmethod
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
  @staticmethod
  def getHiddenLayers(out):
    ary = '['
    cnt = 0
    cns = ''
    
    i = AutoNet.getNumberOfLayers(cnt)
    while(cnt < int(i)):
      cns = AutoNet.getNumberofNodes(cnt,cns)
      cns = AutoNet.getActivationFunction(cnt,cns)
      if((cnt+1) != int(int(i))):
        ary += cns + ','
        cns = ''

      else:
        ary += cns + ']'
      
      cnt += 1
      
    e = eval(ary)
    
    return e


  # Asks the user for the model type.
  @staticmethod
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
      i = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')
      
    return t,m


  # Asks the user for the number of inputs.    
  @staticmethod
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
      i = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')
      
    return n

  
  # Asks the user for the number of outputs.
  @staticmethod
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
      i = sys.stdin.readline().strip()
    os.system('cls' if os.name == 'nt' else 'clear')

    return o


  # Gets the data on number of inputs, outputs, and model type.
  @staticmethod
  def getInputs():
    t = ''
    n = 0
    o = 0
    m = ''
    t , m = AutoNet.getModelType()
    n = AutoNet.getInputCount()
    o = AutoNet.getOutputCount()
   
    return [[t,n]],[[t,o]],m


AutoNet.run()
