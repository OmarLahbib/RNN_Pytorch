from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd

from Data_Preprocessing import Data_Preprocessing
from Dates_Pre import Dates
from Metrics import filling_by_batch , MIOU , transform
from Models import RNNModel ,GRUModel ,LSTMModel

import argparse
#import glob

parser = argparse.ArgumentParser(description='Crops Classifcation using Recrrent Neural Networks')

# GENERAL PARAMETER
#parser.add_argument('--ROOT_PATH', default='./data')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')


# NETWORK PARAMETER
parser.add_argument('--net', default='LSTM', help='Choice of the recurrent neural network')
parser.add_argument('--hidden_dim', default=128, type=int, help='hidden dimension inside the recurrent cell')
parser.add_argument('--layer_dim', default=1, type=int, help='shape of model')

#OPTIMIZER PARAMETER
parser.add_argument('--learning_rate', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
parser.add_argument('--lr_steps', default=140, help='List of epochs where the learning rate is decreased by `lr_decay`')

args = parser.parse_args()

#path = args.ROOT_PATH
#trainlist = []
#for fname in glob.glob(path + '/*.csv'):
#    trainlist.append(fname)
    
                                         
#STEP 1: LOADING DATASET-------------------------------------------------------

data_name = '2016_Optical_25' 
# '2017_Optical_3_ID.csv'

features_train, labels_train, labels_true_train, Id_train, Number_Features = Data_Preprocessing('2016_Optical_25.csv', train=True)
features_test,  labels_test, labels_true_test, Id_test  = Data_Preprocessing('2016_Optical_25.csv', train=False)

unique, counts = np.unique(labels_true_train, return_counts=True)
unique1, counts1 = np.unique(labels_true_test, return_counts=True)
dict(zip(unique1, counts1 + counts))

class train_dataset(Dataset):
    def __init__(self):
        
        self.len = features_train.shape[0]
        self.features  = torch.from_numpy(features_train)
        self.labels    = torch.from_numpy(labels_train)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len
    
class test_dataset(Dataset):
    def __init__(self):
        
        self.len = features_test.shape[0]
        self.features  = torch.from_numpy(features_test)
        self.labels    = torch.from_numpy(labels_test)
        self.Id    = torch.from_numpy(Id_test)
        self.labels_true    = torch.from_numpy(labels_true_test)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.Id[index], self.labels_true[index]

    def __len__(self):
        return self.len
    
train_dataset = train_dataset()
test_dataset  = test_dataset()

# Training settings------------------------------------------------------------

batch_size = args.batch_size
num_epochs = args.num_epochs
n_iters = int(num_epochs * len(train_dataset) / batch_size)

#STEP 2: MAKING DATASET ITERABLE-----------------------------------------------

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          drop_last=True)

#STEP 3: CREATE MODEL CLASS----------------------------------------------------

#STEP 4: INSTANTIATE MODEL CLASS-----------------------------------------------

input_dim = Number_Features # Number of features
hidden_dim = args.hidden_dim
layer_dim = args.layer_dim
output_dim = int(labels_test.max()) + 1 #Number Of classes


if args.net == 'RNN' :
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
elif args.net == 'LSTM' :
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
else : model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

    
if torch.cuda.is_available():
    model.cuda()

#STEP 5: INSTANTIATE LOSS CLASS------------------------------------------------
    
unique, counts = np.unique(labels_train, return_counts=True)
counts = np.power(counts.astype(float), -1/2)
counts_t  = torch.from_numpy(counts / counts.sum()).type(torch.FloatTensor).cuda()

criterion = nn.CrossEntropyLoss(counts_t)

#m = nn.Softmax(dim=1)

#STEP 6: INSTANTIATE OPTIMIZER CLASS-------------------------------------------

learning_rate = args.learning_rate

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

#STEP 7: TRAIN THE MODEL-------------------------------------------------------

seq_dim = Dates('Dates.txt')[1] # Number of steps to unroll (Dates)
iter = 0

for epoch in range(num_epochs):
    
    epoch_pass = True
    Decay_pass = True
    total_Train, correct_Train = 0, 0
    
    for i, (features, labels) in enumerate(train_loader):
        features=features.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)
        if torch.cuda.is_available():
            features = Variable(features.cuda())
            labels = Variable(labels.cuda())
        else:
            features = Variable(features) 
            labels = Variable(labels)
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
    
        # Forward pass to get output
        # outputs.size() --> hidden_dim, output_dim 
        outputs = model(features)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        if (epoch == args.lr_steps) and ( Decay_pass == False ) :
            learning_rate = learning_rate * args.lr_decay
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            Decay_pass = True
            
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
            
        total_Train += labels.size(0)
            
        correct_Train += (predicted.type(torch.LongTensor).cpu() == labels.data.cpu()).sum()
            
        accuracy_Train = 100 * correct_Train / total_Train
        
        iter += 1
        
    # Calculate Accuracy         
    correct = 0
    total = 0    
    if epoch_pass == True and epoch % 25 ==0 :
        # Calculate Accuracy         
        correct = 0
        total = 0
        # Iterate through test dataset
        for features, labels, Id, labels_true in test_loader:
            features=features.type(torch.FloatTensor)
            if torch.cuda.is_available():
                features = Variable(features.cuda())
                
            # Forward pass only to get output
            outputs = model(features)
            
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            
            # Total number of labels
            total += labels.size(0)
            
            # Total correct predictions
            correct += (predicted.type(torch.DoubleTensor).cpu() == labels.cpu()).sum()
        
        accuracy = 100 * correct / total
        
        epoch_pass = False
        
        # Print Loss
        print('Iteration: {}. epoch: {}. Loss: {:.3f}. Accuracy_Train {:.3f}. Accuracy_Test: {:.3f}  '.format(iter, epoch, loss.data[0], accuracy_Train, accuracy))

       
# Calculate Accuracy / IOU / MIOU----------------------------------------------
cm = np.zeros((output_dim,output_dim))
results = np.zeros((len(test_loader)*batch_size,output_dim+3))
predictions = np.zeros((0))
proba = np.zeros((0,output_dim)).astype('float')
Accuracies = np.zeros((output_dim+5))
Ids = []
labels_true_l = []

correct = 0
total = 0           
for features, labels, Id, labels_true  in test_loader:
    features=features.type(torch.FloatTensor)
    if torch.cuda.is_available():
        features = Variable(features.cuda())
                    
        # Forward pass only to get output
        outputs = model(features)
        
        # Get the probability
        prob = F.softmax(outputs, dim=1)
        proba = np.append(proba, prob.data.cpu().numpy(), axis=0)
        
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        predictions = np.append( predictions, predicted.cpu().numpy(), axis=0)

        Ids = np.append( Ids, Id.cpu().numpy(), axis=0)
        labels_true_l = np.append( labels_true_l, labels_true.cpu().numpy(), axis=0)
        # Total number of labels
        total += labels.size(0)
                
        # Confusion Matrix
        if epoch == num_epochs-1 :
            cm = cm + filling_by_batch(labels, torch.max(outputs.data, 1)[1], output_dim)

# getting results--------------------------------------------------------------
Ids= np.array(Ids)
labels_true_l = np.array(labels_true_l)
Fs , Pre, Rec, M = MIOU(cm, output_dim)

unique_t, _ = np.unique(labels_train, return_counts=True)
unique, count = np.unique(labels_true_train, return_counts=True)
OA_3 = 0

for ii in range(0,proba.shape[0]):
    temp = np.argsort(proba[ii,:])[-3:]
    
    for i in range (0,temp.shape[0]):
        for j in range (0,unique_t.shape[0]):
            if ( temp[i] == j ):
                k = j
                break
        temp[i] = int(unique[k])
    
    if ( int(labels_true_l[ii]) in temp[-3:] ):
        OA_3 += 1
        
# Parcel_ID/Ground_Truth/Prediction/Prob for each class------------------------

results[:,0]     = Ids
results[:,1]     = labels_true_l[0:predictions.shape[0]] 
results[:,3:3+output_dim] = proba.astype(float)

a = np.array(predictions)
for i in range(0,predictions.shape[0]):   
    for j in range (0,unique_t.shape[0]):
        if ( predictions[i] == j ):
            k = j
            break
    predictions[i] = int(unique[k])
    
results[:,2]     = predictions


columns = ['Id','Ground_Truth','Prediction']
for i in range(0,output_dim):
    columns = columns + ['Prob {}'.format(int(unique[i]))]

re = pd.DataFrame(data=results,      # values
              index=results[:,0],    # 1st column as index
              columns=columns )      # 1st row as the column names

re['Id'] = re['Id'].astype("int")
re['Ground_Truth'] = re['Ground_Truth'].astype("int")
re['Prediction'] = re['Prediction'].astype("int")

re.to_csv(data_name + '_results.csv', sep='\t', encoding='utf-8',index=False)

# Precison/Recall--------------------------------------------------------------

Pre_Recall = pd.DataFrame(data=np.append(np.transpose([Pre]), np.transpose([Rec]), axis=1),                    
              index=unique.astype(int),                                                      
              columns=['Precision','Recall'] ) 

Pre_Recall.to_csv(data_name + '_Precison_Recall.csv', sep='\t', encoding='utf-8',index=True)

# acuracy_train/Accuracy_test/MIOU/Fsocre--------------------------------------

line = ['Accuracy_Train','Accuracy_Test','MIOU', 'IoU_Pond' , 'Best_3_OA']
IoUp = 0
S=0
for i in range(0,output_dim):
    line = line + ['IoU {}'.format(int(unique[i]))]
    IoUp = IoUp + Fs[i] * count[i]
    S = S+ count[i]
    
Accuracies[0], Accuracies[1], Accuracies[2], Accuracies[3], Accuracies[4], Accuracies[5:] = accuracy_Train, accuracy, M, IoUp / S , (OA_3/proba.shape[0] * 100), Fs

acc = pd.DataFrame(data=[Accuracies],          
                  index=[[1]],       
                  columns=line )            

acc.to_csv(data_name + '_accuracies.csv', sep='\t', encoding='utf-8',index=False)

# Confusin Matrix--------------------------------------------------------------

confusion = pd.DataFrame(data=cm,      
              index=unique.astype(int),    
              columns=unique.astype(int) )      

confusion.to_csv(data_name + '_confusin_matrix.csv', sep='\t', encoding='utf-8',index=True)

# Frequency of each Class------------------------------------------------------
unique, count = np.unique(labels_true_train, return_counts=True)

columns = ['Class','Frequency']

UN = pd.DataFrame(data=np.append(np.transpose([unique]), np.transpose([count]), axis=1),      
              index=unique.astype(int),    
              columns=columns )

UN.to_csv(data_name + '_frequency\class.csv', sep='\t', encoding='utf-8',index=False)












