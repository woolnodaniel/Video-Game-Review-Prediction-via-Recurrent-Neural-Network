#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

"""
ANSWER TO QUESTION: Model Design

The program I have designed is a weighted LSTM/GRU model. The models are made in  the same way
as presented in the lectures. For each input, we pass in one vector (word) at a time,
and then update the hidden states accordingly. This is repeated for each time
step. The number of epochs (20) and the initial learning rate (0.85) were experimentally determined.
A grid search showed that the learning rate achieved the highest weighted score over 20 epochs,
and further tests showed that after about 20, 21 epochs the training error increased,
indicating overfitting. 

A weighted LSTM/GRU was chosen because (a) individually, both models performed roughly the same, and
(b) upon further inspection, they performed well on different inputs. This showed that combining them
would be a winning strategy, as the model could (potentially) learn both strategies and generalise.
Finally we also use a learning rate decay. Experiments showed a large lr was needed initially to lower
the error, but that over time this became unstable. Hence the decay. The multiplicative factor was
again determined experimentally.

Other features of the program:
    o Preprocessing: removing any words that are 2 or less characters in length, as well as
      any special characters.
    o Stopwords: Chose not to use, as I covered most of the important ones with my two
      character rule above, and my weighting of certain types of words.
    o Labels/loss/output: The labels are converted to class indices (i.e. subtract one), and the
      model then aims to increase the probability of the correct class, hence there are 5 outputs.
      For a multi-class prediction problem such as this, the appropriate loss function was
      Cross Entropy. Using a regression approach was more prone to failure, as it did not
      penalise being one star away as much, and predicting one star away too often returns
      a bad weighted average.
"""

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as toptim
import torchtext
from torchtext.vocab import GloVe
import re

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    new_sample = []
    for j in range(len(sample)):
        word = sample[j]
        word = re.sub(r'[^\w]','',word)
        if len(word) > 2:
            new_sample.append(word)

    return new_sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch

stopWords = {}
glove_dim = 50
wordVectors = GloVe(name='6B', dim=glove_dim)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """

    S = datasetLabel.size()[0]
    label_proba = torch.zeros((S,))
    for i in range(S):
        label_proba[i] = int(datasetLabel[i])-1
    return label_proba.long()

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    final_out = torch.tensor([torch.argmax(o)+1 for o in netOutput]).view((netOutput.size()[0],1))
    final = final_out.float()
    final.requires_grad = True
    return final

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    """ Model: LSTM
        Model design as per LSTM:
            o x: vector of dimension input_dim
            o h: vector of dimension hidden_dim
            o c: vector of dimension hidden_dim
        Transition equations:
            1. f = sigmoid(Wf*[x[t]; h[t-1]])
            2. i = sigmoid(Wi*[x[t]; h[t-1]])
            3. g = tanh(Wg*[x[t]; h[t-1]])
            4. o = sigmoid(Wo*[x[t]; h[t-1]])
            5. c[t] <- c[t-1].f + i.g
            6. h[t] <- tanh(c[t]).o
        h[0] and c[0] are initialised to small random weights.
        After computing this for all x[t], the output is given by a linear function
            of the final h, with sigmoid activation.
    """
    def __init__(self):
        super(network, self).__init__()
        self.input_dim = glove_dim
        self.hidden_dim = 4*glove_dim
        self.output_dim = 5

        #for lstm
        self.f = tnn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.i = tnn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.g = tnn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.o = tnn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)

        #for gru
        self.z = tnn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.r = tnn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.ht = tnn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)

        #weightings
        self.alpha = tnn.Linear(self.hidden_dim + self.hidden_dim, 1)

        #outputs
        self.lstm_hid2out = tnn.Linear(self.hidden_dim + self.hidden_dim, self.output_dim)
        self.gru_hid2out = tnn.Linear(self.hidden_dim, self.output_dim)
        self.final_output = tnn.Linear(self.output_dim + self.output_dim, self.output_dim)

        #activations and other
        self.sigmoid = tnn.Sigmoid()
        self.tanh = tnn.Tanh()
        self.drop = tnn.Dropout(0.05)

        #for scheduler
        self.step = False


    def forward(self, input, length):
        #number of layers in the RNN: i.e.
        if self.step:
            scheduler.step()
        else:
            self.step = True
        L = int(max(length))
        batch_size = input.size()[0]
        #initialise weights for hidden units h1, h2 and c
        h1 = torch.randn(batch_size, self.hidden_dim)
        h2 = torch.randn(batch_size, self.hidden_dim)
        c = torch.randn(batch_size, self.hidden_dim)
        self.drop(h1)
        self.drop(h2)
        self.drop(c)
        #lstm equations
        for l in range(L):
            x = input[:,l,:]
            xh = torch.cat((x,h1),dim=1)
            f = self.sigmoid(self.f(xh))
            i = self.sigmoid(self.i(xh))
            g = self.tanh(self.g(xh))
            o = self.sigmoid(self.o(xh))
            #update
            c = c*f + i*g
            h1 = self.tanh(c)*o
        lstm_output = self.sigmoid(self.lstm_hid2out(torch.cat((h1,c),dim=1)))

        #gru equations
        for l in range(L):
            x = input[:,l,:]
            xh = torch.cat((x,h2),dim=1)
            z = self.sigmoid(self.z(xh))
            r = self.sigmoid(self.r(xh))
            xht = torch.cat((x,r*h2),dim=1)
            self.drop(xht)
            ht = self.tanh(self.ht(xht))
            h2 = (1-z)*h2 + z*ht
        gru_output = self.sigmoid(self.gru_hid2out(h2))

        #weighted sum
        output = self.final_output(torch.cat((lstm_output,gru_output),dim=1))
        return output


# class loss(tnn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     You may remove/comment out this class if you are not using it.
#     """
#
#     def __init__(self):
#         super(loss, self).__init__()
#
#     def forward(self, output, target):
#         pass

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
#lossFunc = loss()
lossFunc = tnn.CrossEntropyLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.9
batchSize = 32
epochs = 20
optimiser = toptim.SGD(net.parameters(), lr=0.85)
lambda1 = lambda epoch: 0.9997
scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda1)
