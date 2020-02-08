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
		self.mini_batch_size = mini_batch_size

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

			mini_batches_X, mini_batches_Y = [], []

			for batch in mini_batches:
				mini_batches_X.append(np.column_stack(tuple([batch[k][0] for k in range(len(batch))])))
				mini_batches_Y.append(np.column_stack(tuple([batch[k][1] for k in range(len(batch))])))

			for X, Y in zip(mini_batches_X, mini_batches_Y):
				self.update_mini_batch(X, Y, eta)

			if test_data:
				print("epoch {}: {} / {}".format(j,self.evaluate(test_data),n_test))
			else:
				print("epoch {} done".format(j))


	def update_mini_batch(self,X_mat,y_mat,eta):

		delta_nabla_b, delta_nabla_w = self.backprop(X_mat,y_mat)
		delta_b_avg = [np.dot(bias_grad,np.ones((bias_grad.shape[1],1))) for bias_grad in delta_nabla_b]
		
		self.weights = [w - float(eta/(X_mat.shape[1]))* gw for w,gw in zip(self.weights,delta_nabla_w)]
		self.biases = [b - float(eta/(X_mat.shape[1])) * gb for b,gb in zip(self.biases,delta_b_avg)]


		


	def backprop(self,X_mat,y_mat):
		nabla_b = [np.zeros((b.shape[0],X_mat.shape[1])) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = X_mat
		activations = [X_mat]
		zs = []
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = self.cost_derivation(activations[-1],y_mat) * sigmoid_prime(zs[-1])
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


	
if __name__ == '__main__':

	def sigmoid_prime(z):
		return sigmoid(z)*(1-sigmoid(z))


	def sigmoid(z):
		return 1.0/(1.0+ np.exp(-z))




	training_data,validation_data,testing_data = mnist_loader.load_data_wrapper()

	



	net = Network([784,8,10])
	net.SGD(training_data,epochs=30,mini_batch_size=30,eta=3.0,test_data=testing_data)




	




	
	
