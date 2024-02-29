import numpy as np
from calculations import NeuralNetwork as nn
import os

# List of data
train_data = []
test_data = []

# Datset path
train_dataset_path = os.path.join(os.path.dirname(__file__), 'dataset/digits_dataset.txt')

# Model save path
save_path = os.path.join(os.path.dirname(__file__), 'models/model_digits.txt')

# Read txt file to recieve train dataset
with open(train_dataset_path, 'r') as f:
    for line in f:
        data = line.rstrip('\n').split(' ')
        train_data.append(data)

# list of dataset
train_inputs = []
train_labels = []
 
# Read lines in txt file for train dataset then append to list
for i in range(len(train_data)):
    train_inputs.append(list(map(int,train_data[i][0:25])))
    train_labels.append(list(map(int,train_data[i][25:])))

# Function to create model
def create_model():
    # Add input layer with input shape of 25
    model = nn.addLayer.Input(neurons=25)
    # Add hidden layer with 25 neurons and sigmoid activation function
    model = nn.addLayer.Hidden(neurons=25, model=model, activation='sigmoid')
    # Add output layer with 4 neurons and sigmoid activation function
    model = nn.addLayer.Output(neurons=5, model=model, activation="sigmoid")

    return model

# Function to train the model
def train_model(model, inputs, labels, save_path):
    # Train model
    model.train(inputs=inputs, 
                labels=labels, 
                epoch=1500, 
                learning_rate=5)
    
    # Save the model after train so we can load it again later
    model.saveModel(save_path)

# Function to predict the output from a model
def predict(model, inputs, labels):
    # Predict the output based on the inputs
    pred = model.predict(inputs)

    # Print the results
    acc = 0
    print(f"{'{:^75}'.format('Inputs')}   {'{:^15}'.format('Labels')}   {'{:^15}'.format('Predicts')}")
    for i in range(len(pred)):
        # Finding the highest value
        pred[i] = [1 if j == max(pred[i]) else 0 for j in pred[i]]
        # Check if predicts is equal to labels
        if labels[i] == pred[i]:
            acc += 1
        print(inputs[i], " ",labels[i], " ", pred[i])
    acc = (acc/len(pred))*100
    print(f"\nAccuracy: {round(acc)}%")

# Create model
model = create_model()

# If the model is saved | Use this code below to load the model from the txt file
model = nn.loadModel(model_path=save_path)

# Print the model layers
model.eval()

# Train the model
# train_model(model, train_inputs, train_labels, save_path=save_path)

# Predict the output from train dataset
print(f"\n{' Train Dataset Output ':=^111}")
predict(model, train_inputs, train_labels)

# Self input
while True:
    z = input("\nInput\t: ").strip()
    if z == 'x': break

    z = [int(i) for i in z.split(' ')]

    if len(z) != len(train_inputs[0]):
        print("Input length is not the same as input layer shape")
    else:
        pred = model.predict(np.expand_dims(z, axis=0))
        pred = np.array(pred).squeeze()
        pred = [1 if i == max(pred) else 0 for i in pred]
        print(f"Output\t: {pred}")
        