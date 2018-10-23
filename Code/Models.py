import torch
import torch.nn as nn
from torch.autograd import Variable

#STEP 3: CREATE MODEL CLASS----------------------------------------------------

""" a simple Recurrent Neural Network Model """

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building the RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim , seq_dim = Number of dates , feature_dim = Number of classes)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))        
        
        
        out, hn, = self.rnn(x, h0)
        
        # we just want the last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> [batch_size, Number Of classes]
        return out
    
  
""" a Gated Recurrent Unit Neural Network Model """

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building the Gru
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim , seq_dim = Number of dates , feature_dim = Number of classes)
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))        
        
        
        out, hn, = self.gru(x, h0)
        
        # we just want the last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> [batch_size, Number Of classes]
        return out
    
      
""" a Long Short Term Memory Neural Network Model """        
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building the LSTM
        # batch_first=True causes input/output tensors to be of shape
        # ((batch_dim , seq_dim = Number of dates , feature_dim = Number of classes)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        self.drop = nn.Dropout(0.7)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        
        out, (hn, cn) = self.lstm(x, (h0,c0))
        
        out = self.drop(out)
        
        out = self.fc(out[:, -1, :]) 
        # out.size() --> [batch_size, Number Of classes]
        return out

""" Multi Layered Perceptron """ 

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3 ): #, output_dim
        super(MLPModel, self).__init__()
        # Linear function 1: 
        self.fc1 = nn.Linear(input_dim, hidden_dim_1) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        
        # Linear function 3: 
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        # Non-linearity 3
        self.relu3 = nn.ReLU()
        
        # Linear function 4 (readout): 
        #self.fc4 = nn.Linear(hidden_dim_3, output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        
        # Linear function 3
        out = self.fc3(out)
        
        # Non-linearity 3
        #out = self.relu3(out)
        
        # Linear function 4 (readout)
        #out = self.fc4(out)
        return out
    



































