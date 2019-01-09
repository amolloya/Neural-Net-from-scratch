from PIL import Image
from numpy import array
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from random import randint

# Loading in the data
# Training set
x1 = []
for i in range(1,91):
    img = Image.open('circles\drawing(%s).png' % i)
    arr = array(img)
    arr = arr[:,:,1]
    arr = np.reshape(arr,(1,784))
    x1 = np.append(x1, arr)
x1 = np.reshape(x1,(90,784))
    
x2 = []
for i in range(1,91):
    img = Image.open('squares\drawing(%s).png' % i)
    arr = array(img)
    arr = arr[:,:,1]
    arr = np.reshape(arr,(1,784))
    x2 = np.append(x2, arr)
x2 = np.reshape(x2,(90,784))

x3 = []
for i in range(1,91):
    img = Image.open('triangles\drawing(%s).png' % i)
    arr = array(img)
    arr = arr[:,:,1]
    arr = np.reshape(arr,(1,784))
    x3 = np.append(x3, arr)
x3 = np.reshape(x3,(90,784))

x = np.row_stack((x1,x2,x3))

# Training set labels
y1 = np.zeros((90,1), dtype = 'int64')       # 0's for circles
y2 = np.ones((90,1), dtype = 'int64')        # 1's for squares
y3 = 2*np.ones((90,1), dtype = 'int64')      # 2's for traingles

#Converting the label-vector to a matrix
y_ = np.row_stack((y1,y2,y3))
y = np.array([y_]).reshape(-1)
y = np.eye(3)[y]

print('\nFew examples of training set to visualize the input')
#Plot few examples from training set
for i in range(0,5):
    j = randint(0,x.shape[0]-1)
    print('\nInput Image: ')
    plt.imshow(x[j,:].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.axis("off")
    plt.show()
    if y_[j] == 0:
        print('Label: Circle')
    if y_[j] == 1:
        print('Label: Square')
    if y_[j] == 2:
        print('Label: Triangle')
        
#Preprocessing the data
x[x == 255] = 0
x = x/x.max()
x = np.c_[np.ones((x.shape[0])), x]
    
# Test set
x4 = []
for i in range(91,101):
    img = Image.open('circles\drawing(%s).png' % i)
    arr = array(img)
    arr = arr[:,:,1]
    arr = np.reshape(arr,(1,784))
    x4 = np.append(x4, arr)
x4 = np.reshape(x4,(10,784))
    
x5 = []
for i in range(91,101):
    img = Image.open('squares\drawing(%s).png' % i)
    arr = array(img)
    arr = arr[:,:,1]
    arr = np.reshape(arr,(1,784))
    x5 = np.append(x5, arr)
x5 = np.reshape(x5,(10,784))

x6 = []
for i in range(91,101):
    img = Image.open('triangles\drawing(%s).png' % i)
    arr = array(img)
    arr = arr[:,:,1]
    arr = np.reshape(arr,(1,784))
    x6 = np.append(x6, arr)
x6 = np.reshape(x6,(10,784))

x_test = np.row_stack((x4,x5,x6))

#Preprocessing the data
x_test[x_test == 255] = 0
x_test = x_test/x_test.max()
x_test = np.c_[np.ones((x_test.shape[0])), x_test]

# Test set labels
y4 = np.zeros((10,1), dtype = np.int)       # 0's for circles
y5 = np.ones((10,1), dtype = np.int)        # 1's for circles
y6 = 2*np.ones((10,1), dtype = np.int)      # 2's for circles

#Converting the label-vector to a matrix
y_test_ = np.row_stack((y4,y5,y6))
y_test = np.array([y_test_]).reshape(-1)
y_test = np.eye(3)[y_test]

#y = np.reshape(y, (270,3))
#y_test = np.reshape(y_test, (30,3))

#Parameters for learning
epochs = 100
learning_rate = 0.1
batch_size = 32
lambd = 0.01

class NeuralNet(object):
	#Initialise the inital values for the NN
    def __init__(self):
		#Neural Network Model
        print('\nNeural Network Architecture: ')
        self.inputSize = 785 #Number of pixels in a single image
        self.outputSize = 3 #Output to classify which shape it is
        print('Input layer: ', self.inputSize)
        print('Number of classes (Output layer): ', self.outputSize )
        hiddenSize = int(input('Input the number of neurons in the hidden layer of neural network architecture: '))

        self.hiddenSize = hiddenSize #Number of neurons in hidden layer
		
		#Create the weights randomly into a matrix of the same size as the number of nodes they are connected to 
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #input -> hidden
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #hidden -> output
    
    
    #Function to get mini-batch data for calculating gradients
    def next_batch(self,x, y, batchSize):
        for i in np.arange(0, x.shape[0], batchSize):
            yield(x[i:i + batchSize], y[i:i + batchSize])
            
    
    #Predict function: Use this after the network is trained to predict
    def predict(self, x):
        prediction = self.forwardProp(x)
        return prediction
    
    
    #Propagate the data forward through the network using sigmoid function as the activation function
    def forwardProp(self, x):
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.h_x = self.sigmoid(self.z3)
        return self.h_x
    
    
    #Sigmoid equation for use as the activation function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    
    #Cost function to find out how wrong we are when training
    def costFunction(self, x, y,lambd):
        self.h_x = self.forwardProp(x)
        J = (1/(2*x.shape[0]))*sum(sum((y-self.h_x)**2))
        
        # Compute regularization cost
        regularization_cost = (lambd/(2*x.shape[0]))*(np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
    
        # add cost and regularization_cost
        J = J + regularization_cost
        return J
    
    
    #Derived sigmoid function used in back prop
    def sigmoidDerived(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    
    #Back prop to get the gradients wrt to weights
    def backProp(self, X, y):
        self.h_x = self.forwardProp(X)
		#Weight Layer 2
        delta3 = np.multiply(self.h_x-y, self.sigmoidDerived(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

		#Weight Layer 1
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidDerived(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2
    
    
    #Train your model with Mini-batch gradient
    def train(self,x,y,epochs,batch_size,learning_rate,lambd):
        
        self.J = []
        self.train_error = []
        self.train_precision = []
        self.test_error = []
        self.test_precision = []
        
        for epoch in np.arange(0, epochs):
	        # initialize the total loss for the epoch
            
            cost = []
	        # loop over our data in batches
            for (batchX, batchY) in self.next_batch(x, y, batch_size):   
                
                dJdW1, dJdW2 = self.backProp(batchX, batchY)		      
 
		        #Use the gradient descent and weight decay to update the weight
                self.W1 += - learning_rate * dJdW1 -learning_rate*lambd*self.W1
                self.W2 += - learning_rate * dJdW2 -learning_rate*lambd*self.W2
                
                cost = self.costFunction(batchX,batchY,lambd)
                cost += cost
            self.J.append(cost)
            
            #Training set prediction for each epoch
            train_result = []
            for i in range(0,len(x)):
                train_predict = self.predict(x[i])
                train_ = np.argmax(train_predict)
                train_result.append(train_)
            
            #Training set error for each epoch
            count1 = 0
            for i in range(0,len(x)):
                if train_result[i] != y_[i]:
                    count1 = count1 + 1
                    
            # Test set prediction for each epoch
            test_result = []
            for i in range(0,len(x_test)):
                test_predict = self.predict(x_test[i])
                test_ = np.argmax(test_predict)       
                test_result.append(test_)   
            
            #Test set error for each epoch
            count2 = 0
            for i in range(0,len(x_test)):
                if test_result[i] != y_test_[i]:
                    count2 = count2 + 1
                    
            error1 = (count1/x.shape[0])*100
            self.train_error.append(error1)
            preci1 = 100 - (count1/x.shape[0])*100
            self.train_precision.append(preci1)
            
            error2 = (count2/x_test.shape[0])*100
            self.test_error.append(error2)
            preci2 = 100 - (count2/x_test.shape[0])*100
            self.test_precision.append(preci2)
            
        return self.W1,self.W2, self.J, self.train_error, self.train_precision, self.test_error, self.test_precision
    
#Main function that creates an object from the class and train and predict the results
def pred():
    net = NeuralNet()
    costBefore = float(net.costFunction(x,y,lambd))
    net.train(x,y,epochs,batch_size,learning_rate,lambd)
    costAfter = float(net.costFunction(x,y,lambd))

    print("\nCost Before: " + str(costBefore))
    print("Cost After: " + str(costAfter))
    #print("Cost difference: " + str(costBefore - costAfter))

    print('\nPlot of Cost vs Iterations')
    plt.plot(net.J)
    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    
    print('\nPlot of Train and Test error vs Iterations')
    plt.plot(net.train_error, label='Training error')
    plt.plot(net.test_error, label='Test error')
    plt.grid(1)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
    
    print('\nPlot of Train and Test precision vs Iterations')
    plt.plot(net.train_precision, label='Training precision')
    plt.plot(net.test_precision, label='Test precision')
    plt.grid(1)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Precision')
    plt.show()
    
    #Training set prediction
    train_result = []
    for i in range(0,len(x)):
        train_predict = net.predict(x[i])
        train_ = np.argmax(train_predict)
        train_result.append(train_)
        
    #Training set error
    count1 = 0
    for i in range(0,len(x)):
        if train_result[i] != y_[i]:
            count1 = count1 + 1
    print('\nMisclassified examples in training set (out of 270): ', count1)
    print('Training error rate: ', (count1/x.shape[0])*100)
    print('Training precision: ', 100 - (count1/x.shape[0])*100)
    
    #Test set prediction
    test_result = []
    for i in range(0,len(x_test)):
        test_predict = net.predict(x_test[i])
        test_ = np.argmax(test_predict)       
        test_result.append(test_)   
        
    #Test set error   
    count2 = 0
    for i in range(0,len(x_test)):
        if test_result[i] != y_test_[i]:
            count2 = count2 + 1
    print('\nMisclassified examples in test set (out of 30): ', count2)
    print('Test error rate: ', (count2/x_test.shape[0])*100)
    print('Test precision: ', 100 - (count2/x_test.shape[0])*100)
        
    #Test set: Prediction and actual image
    print('\nFew random examples from test set, their actual labels and predictions:')
    for i in range(0,3):
        j = randint(0,x_test.shape[0]-1)
        print('\nInput Image: ')
        plt.imshow(x_test[j,1:].reshape(28,28), cmap = matplotlib.cm.binary)
        plt.axis("off")
        plt.show()
        if test_result[j] == 0:
            print('Predicted label: Circle')
        if test_result[j] == 1:
            print('Predicted label: Square')
        if test_result[j] == 2:
            print('Predicted label: Triangle')
        
        if test_result[j] == y_test_[j]:
            print('Correctly classified')
        else:
            print('Misclassified')
            
if __name__ == "__main__":
	pred()
