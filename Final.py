# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import time

# define the number of classes, runs, and population size
num_classes = 2
runs = 6
population_size = 200

# load all data
data = pd.ExcelFile('mock_q6_mod.xlsx')# names=names)
data = data.parse("Sheet2", header=None)

# mix true/false values
tmp = np.zeros([data.shape[0], data.shape[1]])
for x in range(0, 15):
    tmp[2*x:2*x+1] = data[x:x+1]
    tmp[2*x+1:2*x+2] = data[x+15:x+16]
tmp[30:31] = data[30:31]
data = tmp

def unit_vector(vector):
    """ return unit vector """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ return the angle between two vectors in degrees"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return 180/np.pi*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# define network
class Net(nn.Module):
    def __init__(self, input_size , hidden_size, num_classes, activationFunction):
        """ initiate instance parameters """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = activationFunction
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.hidden_outputs = 0
        self.output_inputs = 0

    def forward(self, x):
        """ forward pass """
        out = self.fc1(x)
        out = self.tanh(out)
        self.hidden_outputs = out.data
        out = self.fc2(out)
        self.output_inputs = out

        out = self.tanh(out)
        return out

    def get_hidden_outputs(self):
        """ returns the activations of the hidden layer from the last forward pass """
        return self.hidden_outputs

    def find_most_similar(self, angle):
        """ classify hidden neuron pairs by if angle between they're activation over
                last forward pass smaller or larger than input angle, return both sets """
        out = self.get_hidden_outputs().double()
        smallest = angle

        # put neuron pairs into classes based on inner angle
        pos_lower = np.array([])
        pos_upper = np.array([])
        inner_angles = np.array([])
        for i in range(0, out.shape[1]):
            for j in range(i+1, out.shape[1]):
                inner_angle = angle_between(out[:, i], out[:, j])
                inner_angles = np.append(inner_angles, inner_angle)
                if inner_angle < smallest:
                    pos_lower = np.append(pos_lower, np.array([i, j]))
                else:
                    pos_upper = np.append(pos_upper, np.array([i, j]))

        # reshape neuron pairs arrays to be, number of pairs by 2
        if pos_lower.size > 2:
            pos_lower = pos_lower.reshape((int(pos_lower.size/2), 2))
        if pos_upper.size > 2:
            pos_upper = pos_upper.reshape((int(pos_upper.size/2), 2))

        return pos_lower, pos_upper, inner_angles

    def remove_most_similar(self, min_angle):
        """ for neuron pairs with inner angles smaller than input min_angle add output weights of first neuron to second
                and set output weights of first neuron to zero. Return number of neurons no longer used """
        pos, _ , _= self.find_most_similar(min_angle)
        for neuron in pos:
            i = 0
            if pos.size==2:
                if i ==1:
                    continue
                neuron = pos
                i += 1
            self.fc2.weight.data[:, int(neuron[1])] += self.fc2.weight.data[:, int(neuron[0])]
            self.fc2.weight.data[:, int(neuron[0])] = torch.tensor(
                np.zeros([1, self.fc2.weight.data[:, int(neuron[0])].shape[0]]))
        return np.shape(np.where(self.fc2.weight.data[0]==0))[1]

    def remove_complementary_neurons(self, max_angle):
        """ for neuron pairs with inner angles larger than input max_angle subtract output weights of first neuron from
                second and set output weights of first neuron to zero. Return number of neurons no longer used """
        _, pos, _ = self.find_most_similar(max_angle)
        for neuron in pos:
            i = 0
            if pos.size == 2:
                if i == 1:
                    continue
                neuron = pos
                i += 1
            self.fc2.weight.data[:, int(neuron[1])] -= self.fc2.weight.data[:, int(neuron[0])]
            self.fc2.weight.data[:, int(neuron[0])] = torch.tensor(
                np.zeros([1, self.fc2.weight.data[:, int(neuron[0])].shape[0]]))
        return np.shape(np.where(self.fc2.weight.data[0] == 0))[1]


class MyEncoding():
    def __init__(self, identity, lr, epochs, momentum, featuresUsed, layerSizes, activationFunction, lossFunction):
        """ Initialize my solution encoding """
        self.identity = identity
        self.learningRate = lr
        self.epochs = epochs
        self.momentum = momentum
        self.featuresUsed = featuresUsed
        self.layerSizes = layerSizes
        self.activationFunction = activationFunction
        self.lossFunction = lossFunction
        self.accuracy = 0


    def mutate(self, id):
        """ Select mutation from the mutation set d and return a MyEncoding instance with input id as it's identity"""
        mutationType = np.random.randint(7, size=1)[0]
        d = {
            0: self.mutateLearningRate(),
            1: self.reInitialise(id),
            2: self.mutateEpochs(),
            3: self.mutateLayerSizes(),
            4: self.mutateFeaturesUsed(),
            5: self.mutateMomentum(),
            6: self.mutateActivationFunction()
        }
        return d[mutationType]

    def mutateLearningRate(self):
        """ Multiply learning rate by a small factor between 0.5 and 2 and return altered solution encoding """
        factor = 2 ** random.uniform(-1.0, 1.0)
        new_lr = self.learningRate * factor
        return MyEncoding(id, new_lr, self.epochs, self.momentum, self.featuresUsed,
                          self.layerSizes,
                          self.activationFunction, self.lossFunction)

    def reInitialise(self, id):
        """ Randomly re-initialize all hyperparameters based on random uniform values """
        sum1 = 1
        while sum1 < 2:
            featuresUsed = np.random.randint(2, size=(1, 20))
            sum1 = sum(sum(featuresUsed))

        activationFunctionList = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                                  nn.Tanhshrink(), nn.Hardtanh()]
        return MyEncoding(id, 10**(-1*np.random.uniform(0,6)), int(np.random.uniform(1, 100)), np.random.uniform(0, 0.99),
                          featuresUsed[0], int(2+np.random.randint(49, size=(1, 1))[0]),
                          random.choice(activationFunctionList), self.lossFunction)

    def mutateEpochs(self):
        """ Increase or decrease number of epochs by 1 and return altered solution encoding """
        new_epochs = np.random.randint(3, size=1)-1 + self.epochs
        if new_epochs<1:
            new_epochs = 1
        elif new_epochs>100:
            new_epochs = 100

        return MyEncoding(id, self.learningRate, int(new_epochs), self.momentum, self.featuresUsed,
                          self.layerSizes, self.activationFunction, self.lossFunction)

    def mutateLayerSizes(self):
        """ Increase or decrease number of hidden neurons by 1 and return altered solution encoding """
        new_layerSizes = np.random.randint(3, size=1) - 1 + self.layerSizes
        if new_layerSizes < 2:
            new_layerSizes = 2
        elif new_layerSizes > 50:
            new_layerSizes = 50
        return MyEncoding(id, self.learningRate, self.epochs, self.momentum, self.featuresUsed,
                          int(new_layerSizes), self.activationFunction, self.lossFunction)

    def mutateFeaturesUsed(self):
        """ XOR one of the features used where 1 indicates the feature is used and 0
        indicates it is not and return altered solution encoding """
        new_featuresUsed = self.featuresUsed+1
        new_featuresUsed = new_featuresUsed-1
        ind = np.random.randint(20, size=1)[0]
        new_featuresUsed[ind] = (new_featuresUsed[ind]+1)%2

        return MyEncoding(id, self.learningRate, self.epochs, self.momentum, new_featuresUsed,
                      self.layerSizes, self.activationFunction, self.lossFunction)

    def mutateMomentum(self):
        """ Multiply momentum by a small factor between 0.5 and 2 and return altered solution encoding """
        factor = 2 ** random.uniform(-1.0, 1.0)
        new_momentum = self.momentum * factor
        return MyEncoding(id, self.learningRate, self.epochs, new_momentum, self.featuresUsed,
                          self.layerSizes, self.activationFunction, self.lossFunction)

    def mutateActivationFunction(self):
        """ Randomly re-select an activation from possible list and return altered solution encoding """
        activationFunctionList = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                                  nn.Tanhshrink(), nn.Hardtanh()]
        new_activationFunction = random.choice(activationFunctionList)
        return MyEncoding(id, self.learningRate, self.epochs, self.momentum, self.featuresUsed,
                          self.layerSizes, new_activationFunction, self.lossFunction)


def cross_train(encoding, runs):
    """ Takes the solution encoding and number of runs and returns a single value for the accuracy of the solution.
    The accuracy is the average 5-fold validation accuracy over the number of runs, which was set at 6."""
    total_test = 0
    correct_test = 0
    for n in range(runs):
        # cross validation
        for x in range(5):
            # split data into training set (25) and testing set (6)
            msk = np.ones((data.shape[0]), dtype=bool)
            msk[x*6:x*6+6] = False
            data_shuffled = data[torch.randperm(data.shape[0])]
            train_data = data_shuffled[msk]
            test_data = data_shuffled[~msk]

            # remove features which are not used
            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
            train_data = np.delete(train_data, np.s_[get_indexes(0, encoding.featuresUsed)], axis=1)
            test_data = np.delete(test_data, np.s_[get_indexes(0, encoding.featuresUsed)],axis=1)

            # split training data into input and target
            train_input = torch.tensor(train_data[:, :int(sum(encoding.featuresUsed))]).float()
            train_target = torch.tensor(train_data[:, int(sum(encoding.featuresUsed))]).long()
            
            # split training data into input and target
            test_input = torch.tensor(test_data[:, :int(sum(encoding.featuresUsed))]).float()
            test_target = torch.tensor(test_data[:, int(sum(encoding.featuresUsed))]).long()
            
            # initialize nn, loss function and optimizer
            net = Net(int(sum(encoding.featuresUsed)), encoding.layerSizes, num_classes, encoding.activationFunction)
            criterion = encoding.lossFunction
            optimizer = optim.SGD(net.parameters(), lr=encoding.learningRate, momentum=encoding.momentum)
            angles = []

            # iterate through epochs 
            for epoch in range(0, encoding.epochs):
                
                # forward + backward + optimize
                outputs = net(train_input)
                loss = criterion(outputs, train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # test the neural network using testing data
            test_output = net(test_input)

            # calculate accuracy
            _, predicted_test = torch.max(test_output, 1)
            total_test = predicted_test.size(0) + total_test
            correct_test = sum(predicted_test.data.numpy() == test_target.data.numpy()) + correct_test
    print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

    return correct_test / total_test


# initialize population
populationDict = {}
for i in range(population_size):

    # initialize learning rate - range [10^0, 10^-6]
    learning_rate = 10**(-1*np.random.uniform(0,6))

    # initialize epochs - range [1, 100]
    epochs = int(np.random.uniform(1, 100))

    # initialize momentum - range [0, 0.99]
    momentum = np.random.uniform(0, 0.99)

    # initialize features used - range [2, 20]
    sum1=1
    while sum1<2:
        featuresUsed = np.random.randint(2, size=(1, 20))
        sum1 = sum(sum(featuresUsed))
    featuresUsed = featuresUsed[0]

    # initialize layerSizes - range [2, 50]
    layerSizes = int(2+np.random.randint(49, size=(1, 1))[0])

    # initialize activationFunction
    activationFunctionList = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(),
                             nn.Tanhshrink(), nn.Hardtanh()]
    activationFunction = random.choice(activationFunctionList)

    # initialize lossFunction
    lossFunction = nn.CrossEntropyLoss()

    populationDict[i] = MyEncoding(i, learning_rate,epochs, momentum, featuresUsed, layerSizes,
                                activationFunction, lossFunction)

    populationDict[i].accuracy = cross_train(populationDict[i], runs, False)


# repeatedly compare, kill, replace killed with mutations of stronger
# solution for number of iterations specified in fights
fights = 1000000
for i in range(fights):
    # select fighters
    fighters = random.sample(range(0, population_size), 2)
    # compare fitness
    if populationDict[fighters[0]].accuracy > populationDict[fighters[1]].accuracy:
        parent = fighters[0]
        child = fighters[1]
    else:
        parent = fighters[1]
        child = fighters[0]

    # replace weaker solution with mutation of stronger solution
    populationDict[child] = populationDict[parent].mutate(child)
    populationDict[child].accuracy = cross_train(populationDict[child], runs, False)

    # occasionally display accuracy reached and solution values
    if i % 50 == 0:
        mostAccurate = 0
        maxAccuracy = 0
        for j in range(population_size):
            if populationDict[j].accuracy > maxAccuracy:
                maxAccuracy = populationDict[j].accuracy
                mostAccurate = j
        print("Most Accurate: ", populationDict[mostAccurate].accuracy," After: ", i, " fights")
        attrs = vars(populationDict[mostAccurate])
        print(', '.join("%s: %s" % item for item in attrs.items()))

        # re-evaluate most accurate to prevent fluke repeatedly killing competitors
        populationDict[mostAccurate].accuracy = cross_train(populationDict[mostAccurate], runs*2, False)
