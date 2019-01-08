import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import randint

#Function to load MNIST dataset
def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([nImages, width, height])

    labels = np.fromfile(folder + "/" + prefix + '-labels-idx1-ubyte', dtype = 'ubyte')[2 * intType.itemsize:]
    return data, labels

#Function to convert the labels from (m x 1) to (m x 10) array with zeroes elsewhere except at the correct label
def toHotEncoding(classification):
    hotEncoding = np.zeros([len(classification), np.max(classification) + 1 ])
    hotEncoding[np.arange(len(hotEncoding)), classification] = 1
    return hotEncoding

#load training and testing images and labels       
training_images, training_labels = loadMNIST( "train", "mnist-dataset/" )
test_images, test_labels = loadMNIST( "t10k", "mnist-dataset/" )

# Normalize the data
training_images = training_images/training_images.max()
test_images = test_images/test_images.max()

#Converting image pixels from (m x 28 x 28) pixels to (m x 784) to feed into ANN
training_images_ = training_images.reshape(training_images.shape[0], 784)
test_images_ = test_images.reshape(test_images.shape[0], 784)

# Dimension of Training and Testing set
print('\nNumber of examples in training set:', training_images_.shape[0])
#print('\n1st row', x[0])
print('Number of examples in test set:', test_images_.shape[0])

#Converting the label-vector to a matrix 
training_labels_ = toHotEncoding(training_labels)
test_labels_ = toHotEncoding(test_labels)

#Plot few examples from training set
for i in range(0,5):
    print('\nInput Image: ')
    plt.imshow(training_images_[i,:].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.axis("off")
    plt.show()
    print('Label: ', training_labels[i])

training_images_ = np.append(np.ones((training_images_.shape[0], 1)), training_images_, axis = 1)
test_images_ = np.append(np.ones((test_images_.shape[0], 1)), test_images_, axis = 1)
training_labels = np.reshape(training_labels, (training_labels.shape[0],1))
test_labels = np.reshape(test_labels, (test_labels.shape[0],1))

#Parameters for learning
epochs = 100
learning_rate = 0.1
batch_size = 128
lambd = 0.01

class NeuralNet(object):
	#Initialise the inital values for the NN
    def __init__(self):
		#Neural Network Model
        print('\nNeural Network Architecture: ')
        self.inputSize = 785 #Number of pixels in a single image
        self.outputSize = 10 #Output to classify which digit it is
        print('Input layer: ', self.inputSize)
        print('Number of classes (Output layer): ', self.outputSize )
        hiddenSize1 = int(input('Input the number of neurons in first hidden layer: '))
        hiddenSize2 = int(input('Input the number of neurons in second hidden layer: '))
        print('\nTraining the neural network...')

        self.hiddenSize1 = hiddenSize1 #Number of neurons in hidden layer 1
        self.hiddenSize2 = hiddenSize2 #Number of neurons in hidden layer 2
		
		#Create the weights randomly into a matrix of the same size as the number of nodes they are connected to 
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize1)       #input -> hidden1
        self.W2 = np.random.randn(self.hiddenSize1,self.hiddenSize2)      #hidden1 -> hidden2
        self.W3 = np.random.randn(self.hiddenSize2, self.outputSize)      #hidden2 -> output
    
    
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
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        self.h_x = self.sigmoid(self.z4)
        return self.h_x
    
    
    #Sigmoid equation for use as the activation function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    
    # Cost function to find out how wrong we are when training
    def costFunction(self, x, y, lambd):
        self.h_x = self.forwardProp(x)
        J = (1/(2*x.shape[0]))*sum(sum((y-self.h_x)**2))
        
        # Compute regularization cost for penlizing the weights and weight decay
        regularization_cost = (lambd/(2*x.shape[0]))*(np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
    
        # Add cost and regularization_cost
        J = J + regularization_cost
        return J
    
    # Derived sigmoid function used in back prop
    def sigmoidDerived(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    
    # Back prop to get the gradients wrt to weights
    def backProp(self, X, y):
        #Weight Layer 3
        self.h_x = self.forwardProp(X)
        delta4 = np.multiply(self.h_x-y,self.sigmoidDerived(self.z4))
        dJdW3 = np.dot(self.a3.T, delta4)
        
		# Weight Layer 2
        delta3 = np.dot(delta4,self.W3.T)*self.sigmoidDerived(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)

		# Weight Layer 1
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidDerived(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2, dJdW3
    
    # Train your model with Mini-batch gradient
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
                
                dJdW1, dJdW2, dJdW3 = self.backProp(batchX, batchY)		      
 
		        #Use the gradient descent and weight decay to update the weight
                self.W1 += -learning_rate * dJdW1 -learning_rate*lambd*self.W1
                self.W2 += -learning_rate * dJdW2 -learning_rate*lambd*self.W2
                self.W3 += -learning_rate * dJdW3 -learning_rate*lambd*self.W3
                
                cost = self.costFunction(batchX,batchY,lambd)
                cost += cost
            self.J.append(cost)
                
            #Training set prediction for each epoch
            train_result = []
            for i in range(0,len(training_images_)):
                train_predict = self.predict(training_images_[i])
                train_ = np.argmax(train_predict)
                train_result.append(train_)
                    
            #Training set error for each epoch
            count1 = 0
            for i in range(0,len(training_images_)):
                if train_result[i] != training_labels[i]:
                    count1 = count1 + 1
                        
            # Test set prediction for each epoch
            test_result = []
            for i in range(0,len(test_images_)):
                test_predict = self.predict(test_images_[i])
                test_ = np.argmax(test_predict)       
                test_result.append(test_)  
                
            #Test set error for each epoch
            count2 = 0
            for i in range(0,len(test_images_)):
                if test_result[i] != test_labels[i]:
                    count2 = count2 + 1
        
            error1 = (count1/training_images_.shape[0])*100
            self.train_error.append(error1)
            preci1 = 100 - (count1/training_images_.shape[0])*100
            self.train_precision.append(preci1)
                
            error2 = (count2/test_images_.shape[0])*100
            self.test_error.append(error2)
            preci2 = 100 - (count2/test_images_.shape[0])*100
            self.test_precision.append(preci2)
                    
        return self.W1,self.W2,self.W3, self.J, self.train_error, self.train_precision
    
#Main function that creates an object from the class and train and predict the results
def pred():
    net = NeuralNet()
    costBefore = float(net.costFunction(training_images_,training_labels_,lambd))
    net.train(training_images_,training_labels_,epochs,batch_size,learning_rate,lambd)
    costAfter = float(net.costFunction(training_images_,training_labels_,lambd))

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
    for i in range(0,len(training_images_)):
        train_predict = net.predict(training_images_[i])
        train_ = np.argmax(train_predict)
        
        train_result.append(train_)
        
    #Training set error    
    count1 = 0
    for i in range(0,len(training_images_)):
        if train_result[i] != training_labels[i]:
            count1 = count1 + 1
    print('\nMisclassified examples in training set (out of 60000): ', count1)
    print('Training error rate: ', (count1/training_images_.shape[0])*100)
    print('Training precision: ', 100 - (count1/training_images_.shape[0])*100)
    
    #Test set prediction
    test_result = []
    for i in range(0,len(test_images_)):
        test_predict = net.predict(test_images_[i])
        test_ = np.argmax(test_predict)
        
        test_result.append(test_)   
        
    #Test set error   
    count2 = 0
    for i in range(0,len(test_images_)):
        if test_result[i] != test_labels[i]:
            count2 = count2 + 1
    print('\nMisclassified examples in test set (out of 10000): ', count2)
    print('Test error rate: ', (count2/test_images_.shape[0])*100)
    print('Test precision: ', 100 - (count2/test_images_.shape[0])*100)
    
    #Test set: Prediction and actual image
    print('\nFew random examples from test set and their prediction:')
    for i in range(0,3):
        j = randint(0,test_images_.shape[0])
        print('\nInput Image: ')
        plt.imshow(test_images_[j,1:].reshape(28,28), cmap = matplotlib.cm.binary)
        plt.axis("off")
        plt.show()
        print('Predicted label by the model: ', test_labels[j])
        
        if test_result[j] == test_labels[j]:
            print('Correctly classified')
        if test_result[j] != test_labels[j]:
            print('Misclassified')
        
if __name__ == "__main__":
	pred()
