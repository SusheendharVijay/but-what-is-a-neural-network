import numpy as np 
import mnist_loader
import random 


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
		self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)] 
		


	def backprop(self,x,y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		zs = []
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = self.cost_derivation(activations[-1],y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta,activations[-2].transpose())

		for l in range(2,self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
			nabla_b[-l]  = delta 
			nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())

		return (nabla_b,nabla_w)




	def cost_derivation(self,output_activations,y):
		return (output_activations - y)

	def evaluate(self,testing_data):
		predictions  = [(np.argmax(self.feedforward(x)),y) for x,y in testing_data]
		return sum([int(x==y) for (x,y) in predictions]) 


	


def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))


def sigmoid(z):
	return 1.0/(1.0+ np.exp(-z))




training_data,validation_data,testing_data = mnist_loader.load_data_wrapper()
net = Network([784,5,10])

net.SGD(training_data,90,30,3,test_data=testing_data)

