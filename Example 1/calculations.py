'''Made By: Zikra Fathirizqi Aknes'''

import numpy as np
from loading import Loading

L = Loading()
start_load = L.start_loading_bar
iter_load = L.iterate_loading_bar

activation_list = ['sigmoid']

class NeuralNetwork():
    class addLayer():
        # Function to set input layer
        def Input(neurons):
            model = []
            # Apeend input layer
            model.append(np.random.rand(neurons))

            return model

        # Function to add hidden layer
        def Hidden(neurons, model, activation):
            if not activation in activation_list:
                raise ValueError(f"Activation for '{activation}' is unknown")

            # Append random weights
            model.append(np.random.rand(len(model[-1]), neurons))
            # Append random theta
            model.append(np.random.rand(neurons))
            # Set activation function for this layer
            model.append(activation)
            # Append hidden layer
            model.append(np.random.rand(neurons))

            return model

        # Function to set output layer
        def Output(neurons, model, activation):
            if not activation in activation_list:
                raise ValueError(f"Activation for '{activation}' is unknown")

            # Append random weights
            model.append(np.random.rand(len(model[-1]), neurons))
            # Append random theta
            model.append(np.random.rand(neurons))
            # Set activation function for this layer
            model.append(activation)
            # Append output layer
            model.append(np.random.rand(neurons))

            return Model(model)
    
    # Load model from a txt file
    def loadModel(model_path):    
        model = []

        with open(model_path, 'r') as f:
            for line in f:
                words = line.split(' ')
                if words[0].strip('\n') not in activation_list:
                    model.append(list(map(float, words)))
                else:
                    model.append(words[0].strip('\n'))

        idx = 0
        while idx < len(model)-1:
            j = len(model[idx])
            model[idx+1:idx+1+j] = np.expand_dims(model[idx+1:idx+1+j], 0)
            idx += 4

        return Model(model)
    
class Model():
    def __init__(self, model):
        self.weights = []
        self.theta = []
        self.layers = []
        self.af = []

        self.setVariables(model)

    # Function to train the model
    def train(self, inputs, labels, epoch, learning_rate):
        print("\nTRAINING:")
        # Set the learning rate
        self.lr = learning_rate

        # Iterate for the number of epoch | start_load is to start the loading bar
        for i in range(start_load(epoch)):
            avg_acc = 0
            avg_loss = 0
            # Iterate for the number of inputs
            for j in range(len(inputs)):
                # Set the input layer as the each input that's given
                self.layers[0] = inputs[j]

                # Feed Forward to achieve output | Loop for the number of hidden and output layer 
                for k in range(len(self.layers) - 1):
                    # useAF set to true to achieve the output of the layers
                    self.layers[k+1] = self.FF(self.layers[k], self.layers[k+1], self.weights[k], self.theta[k], useAF=True)
                    
                # Get the value of the output layer
                Y = self.layers[-1]

                # Compute error
                J, E = self.computeError(Y, labels[j])

                # Back Propagation for output layer
                d = self.BP1(E)
                # Loop for the number of hidden layers
                for i in range(2, len(self.weights) + 1):
                    # Back Propagation for hidden layer
                    d = self.BP2(-i, d)
                
                # Add the accuracy and loss for each inputs
                avg_acc += self.accuracy(labels[j], Y)
                avg_loss += J

            # Update the learning rate if its exceed 0.5
            if self.lr > 0.5:
                self.lr *= 0.95

            # Calculate average accuracy and loss by dividing it with the number of inputs 
            avg_acc = avg_acc/len(inputs)
            avg_loss = avg_loss/len(inputs)

            # Iterate the loading bar
            iter_load(avg_loss, avg_acc)

    # Function to predict output
    def predict(self, input):
        # Create a list to store outputs
        outputs = []
        # Iterate for the number of inputs
        for i in range(len(input)):
            # Set the input layer as the each of the given input
            self.layers[0] = input[i]

            # Feed Forward to achieve the output | useAF set to true to achieve the output of the layers
            for k in range(len(self.layers) - 1):
                self.layers[k+1] = self.FF(self.layers[k], self.layers[k+1], self.weights[k], self.theta[k], useAF=True)

            # Append the value from the output layer to the list
            outputs.append(self.layers[-1])

        return outputs

    # Function to calculate sigmoid
    def sigmoid(self, S):
        return 1/(1 + np.exp(-S))
    
    # Function to calculate accuracy
    def accuracy(self, label, output):
        count = 0
        for i in range(len(output)):
            if i == np.argmax(output):
                if label[i] == 1:
                    count += 1
            else:
                if label[i] == 0:
                    count += 1

        return count/len(output)

    # Function to calculate the Feed Forward
    def FF(self, layer1, layer2, weights, theta, useAF):
        # List of strength
        S = []

        # Calculate strength
        for j in range(len(layer2)):
            S.append(0)
            for i in range(len(layer1)):
                S[j] += layer1[i] * weights[i][j]
            S[j] -= theta[j]

            # If useAF is true | Apply activation function
            if useAF:
                # Apply sigmoid to all the list of strength
                S[j] = self.sigmoid(S[j])

        return S

    # Function to calulate error
    def computeError(self, output, label):
        # List of error
        E = []
        # Cost J
        J = 0

        for k in range(len(output)):
            # Calculate and append error
            E.append(label[k] - output[k])
            # Caculate cost J
            J += E[k]**2
        J /= 2

        # Return value of Cost and Error
        return J, E
        
    # Function to calculate back propagation for output layer
    def BP1(self, E):
        # Set the previous layer as H
        H = self.layers[-2]
        # List containing [E*(-1)*d_af] on each value of k
        dk = []
        
        # Compute strength of the last layer | useAF set to false to achieve the strength of neurons
        S = self.FF(self.layers[-2], self.layers[-1], self.weights[-1], self.theta[-1], useAF=False)

        # Iterate for the number of the current hidden layer neurons
        for k in range(len(self.layers[-1])):
            # Calculate derivative of the sigmoid activation function
            d_af = np.exp(-S[k])/(1 + np.exp(-S[k]))**2

            # Calculate and append dk
            dk.append(E[k] * (-1) * d_af)

            # Iterate for the number of the previous layer neurons
            for j in range(len(self.layers[-2])):
                # Update weights
                self.weights[-1][j][k] -= (dk[k] * H[j])*self.lr
            # Update theta
            self.theta[-1][k] -= (dk[k] * (-1))*self.lr

        # Return the value of dk to be used on the next BP calculations
        return sum(dk)

    # Function to calculate back propagation for hidden layers
    def BP2(self, idx, dk):
        # Set the previous layer as x
        x = self.layers[idx-1]
        # List containing [wjk * d_af] on each value of j
        dj = []

        # Compute strength of the current layer | useAF set to false to achieve the strength of neurons
        S = self.FF(self.layers[idx-1], self.layers[idx], self.weights[idx], self.theta[idx], useAF=False)

        # Iterate for the number of the current hidden layer neurons
        for j in range(len(self.layers[idx])):
            # Calculate wjk
            wjk = 0
            for k in range(len(self.layers[idx+1])):
                wjk += self.weights[idx+1][j][k]

            # Calculate derivative of the sigmoid activation function
            d_af = np.exp(-S[j])/(1 + np.exp(-S[j]))**2

            # Calculate and append dj
            dj.append(wjk * d_af)

            # Iterate for the number of the previous layer neurons
            for i in range(len(self.layers[idx-1])):
                # Update weights
                self.weights[idx][i][j] -= (dk * dj[j] * x[i])*self.lr
            # Update theta
            self.theta[idx][j] -= (dk * dj[j] * (-1))*self.lr

        # Return the value of dj to be used on the next BP calculations
        return (dk * sum(dj))

    # Function to set the variables inside the class from the model given
    def setVariables(self, model):
        self.weights = []
        self.theta = []
        self.layers = []
        self.af = []

        for i in range(0, len(model), 4):
            self.layers.append(model[i])

        for i in range(1, len(model), 4):
            self.weights.append(model[i])
        
        for i in range(2, len(model), 4):
            self.theta.append(model[i])
            
        for i in range(3, len(model), 4):
            self.af.append(model[i])

    # Function to save the model to a txt file
    def saveModel(self, save_path):
        model = []
        total_len = len(self.layers) + len(self.af) + len(self.weights) + len(self.theta)

        idx = 0
        for i in range(total_len):
            if idx == 0:
                model.append(list(map(str, self.layers[i//4])))
                idx += 1
            elif idx == 1:
                for data in self.weights[(i-1)//4]:
                    model.append(list(map(str, data)))
                idx += 1
            elif idx == 2:
                model.append(list(map(str, self.theta[(i-2)//4])))
                idx += 1
            elif idx == 3:
                model.append([self.af[(i-3)//4]])
                idx = 0

        with open(save_path, 'w') as f:
            f.write('\n'.join([' '.join(i) for i in model]))

    # Function to print the layers of the model given
    def eval(self):
        print("================= Model Layers =================")
        print("|     Layers     |   Neurons   |   Activation  |")
        print("================================================")
        print(f"|{'Input Layer':^16}|{len(self.layers[0]):^13}|{'-':^15}|")
        for i in range(len(self.layers[1:-1])):
            print(f"|{f'Hidden Layer {i+1}':^16}|{len(self.layers[i+1]):^13}|{self.af[i]:^15}|")
        print(f"|{'Output Layer':^16}|{len(self.layers[-1]):^13}|{self.af[-1]:^15}|")
        print("================================================")
    