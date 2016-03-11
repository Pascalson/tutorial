import theano
import theano.tensor as T
from itertools import izip
import numpy
rng = numpy.random
import pickle

path = "../../wyc2010/DNN_practice/data/"
D = pickle.load( open(path+"dev","r") )
inputs = D[:,1:]
label = D[:,0]

targets = numpy.zeros( ( label.shape[0], 48 ) )
for i in range( label.shape[0] ):
	targets[i][label[i]-1] = 1
#one_hot = T.set_subtensor( z[T.arange(idx.shape[0]), idx], 1 )
#hh = theano.function([idx], one_hot)

#targets = hh( label )

class DNN():
	def __init__(self,layer):
		self.w = []
		self.b = []
		for i in range( len(layer)-1 ):
			self.w.append( theano.shared( rng.randn(layer[i],layer[i+1]) ) ) 
			self.b.append( theano.shared(0.) )
		x = T.matrix() # batch * dim69
		target = T.matrix() # batch * dim48
		output = self.build_DNN( x, layer )
		update_w, update_b = self.update( output, target )
		self.predict = theano.function(
			inputs=[x],
			outputs= ( output > 0.5 )
			)
		self.train = theano.function(
			inputs=[x,target],
			updates = update_w + update_b
			)

	def build_DNN(self,inputs,layer):
		net = []
		net.append( inputs )
		for i in range( 1, len(layer) ):
			net.append( 1 / ( 1 + T.exp( -T.dot(net[i-1], self.w[i-1])-self.b[i-1] ) ) )
		return net[-1]

	def update( self,outputs,target ):
		cost = (-target*T.log(outputs)-(1-target)*T.log(1-outputs)).mean() + 0.01*(self.w[-1]**2).sum()
		dw = T.grad( cost, self.w )
		db = T.grad( cost, self.b )
		lr = 0.01
		param_w = [ ( p, p - lr * q ) for p,q in izip( self.w, dw ) ]
		param_b = [ ( m, m - lr * n ) for m,n in izip( self.b, db ) ]
		return param_w, param_b

max_epoch = 50
batch_num = 200
layer = [69,512,48] #inpur,hidden,hidden,output
#inputs = rng.randn(6)
nn = DNN(layer)
#print( nn.predict(inputs) )
for e in range(max_epoch):
	for b in range(batch_num):
		batch = inputs[64*b:64*(b+1)]
		target = targets[64*b:64*(b+1)]
		nn.train( batch, target )
corr = 0
for i in range( batch_num ):
	seq = nn.predict( inputs[64*i:64*(i+1)] )
	for a in range( 64 ):
		for j in range( 48 ):
			if( seq[a][j] == targets[64*i+a][j] ):
				corr = corr + 1
print( corr )
