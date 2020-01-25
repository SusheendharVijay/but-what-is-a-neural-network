import numpy as np 

class Network(object):
	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

	def feedforward(self,a):
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a) + b)
		return a 

	def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
		if test_data : n_test = len(test_data)
		n = len(training_data)

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)

			if test_data:
				print("epoch {}: {} / {}".format(j,self.evaluate(test_data),n_test))
			else:
				print("epoch {} done".format(j))


	def update_mini_batch(self,mini_batch,eta):

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]


		for x,y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x,y)
			nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
			nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
		self.weights = [w- (eta/len(mini_batch))*nbw for w,nbw in zip(self.weights,nabla_w)]
		self.biases = [b - (eta/len(mini_batch))*nb for b,nbb in zip(self.biases,nabla_b)] 
		


	





def sigmoid(z):
	return 1.0/(1.0+ np.exp(-z))