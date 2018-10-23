import math
import numpy as np
from Dates_Pre import Dates
from sklearn.cross_validation import train_test_split

#Time Parameters
ND=int(Dates('Dates.txt')[1])
#Bandwidth numbers / Radar Channels VV , VH , VV/VH
#B = 10 
#Number of features  (means + variances) * bandwidth / (means + variances) * 3
#F = 20
#test_size splitting the data
test_size = 0.50

#------------------------------------------------------------------OPTICAL DATA
#--- Eliminating Parcels with NaN values -Small Parcels------------------------
def Cleaning(Data,D_x,D_y,F):
    k=[]
    
    for i in range(0,D_x):
        for j in range(0,D_y-2):
            if (math.isnan(Data[i,j])) or (int(Data[i,ND*F]) == 8) or (int(Data[i,ND*F]) == 19) :
                k = np.append( k , [i], axis=0)
                break
            
    return np.delete(Data, (k), axis=0) , len(k)

#--- Filling my Traing data with features from all parcels for all dates-------
    
def Matrix2Tensor(Data, D_x, D_y, ND, k, B, F):
    X= np.vstack(
        tuple(
            [
                np.expand_dims(
                    np.array(
                        [
                            Data[j,i:i+F]
                            for i in range (0,ND*F,F)
                        ]
                    ),
                    axis=0
                )
                for j in range ( 0 , D_x-k )
            ]
        )
    )
                    
    return X

#--- Filling my Traing data with features from all parcels for all dates-------(used for site 04 and 77)
def Filling(Data,D_x,D_y,ND,k,B,F):
    X = np.zeros((0 , ND , F))
      
    for j in range ( 0 , D_x-k ):
        training_set = Data[j:j+1,:]
        Temp = []
        for i in range (0,(D_y-4)//2,B):
            Temp.append( np.concatenate((training_set[0,i:i+B] , training_set[0,i+ND*B:i+ND*B+B]) , axis=0 ))
        Temp= np.array(Temp)
        X = np.append( X, [Temp], axis=0)    
                 
    return X

#--- Feature Scaling-----------------------------------------------------------
def Scalling(Data,ND,F):
    X = Data 
                                
    for i in range (0,ND):
        for j in range ( 0 , F ):
            X[:,i,j] = (Data[:,i,j] - Data[:,i,j].mean()) / Data[:,i,j].std()
            
    return X

#--- Creating the target Array ------------------------------------------------
def Labelling(data,D_x,ND,k,F):
    Y = np.zeros((D_x-k))
    
    for i in range ( 0 , D_x-k ):
        Y[i] =  data[i,ND*F] 
        
    return Y

#--- Creating the true target Array -------------------------------------------
def Labelling_RNN(data,D_x,ND,k,F):
    Y = np.zeros((D_x-k))
    unique, counts = np.unique(data[:,ND*F], return_counts=True)
    
    for i in range ( 0 , D_x-k ):
        for j in range (0 , len(counts)):
            if (data[i,ND*F] == unique[j]):
                k = j           
        if ( int(data[i,ND*F]) != k ):
            Y[i] =  k
        else : Y[i] = int(data[i,ND*F])
        
    return Y

#--- Creating the target Array ------------------------------------------------
def IDing(data,D_x,ND,k,F):
    Y = np.zeros((D_x-k))
    
    for i in range ( 0 , D_x-k ):
        Y[i] =  int(data[i,-1])
        
    return Y

#--- Adding dates to the scaled Data-------------------------------------------
def TimeStamping(Data,D_x,dates,ND,F):
    X = np.zeros((D_x , ND , F+5))
    for i in range ( 0 , D_x ):
        X[i,:,:] =  np.append( Data[i,:,:] , dates, axis=1)
        
    return X        
        
#--- Main ---------------------------------------------------------------------               
def Data_Preprocessing(file_name, train):
    #file_name='Site_04_OPTIC.csv' 'attributs04_2016_R.csv'  '2016_Optical_3_ID.csv' '2016_Optical_25_ID.csv' '2017_Optical_3.csv' '2016_Optical_25.csv'
    data_raw = np.loadtxt(file_name, delimiter=',')
    
    ND=int(Dates('Dates.txt')[1])
    dates=Dates('Dates.txt')[0]
    
    D_x=data_raw.shape[0]
    D_y=data_raw.shape[1]
    
    #Parameters selection
    if ((D_y - 2)//ND == 20) :
        B = 10
        F = 20
        input_dim = F
    else :
        B = 3
        F = 6
        input_dim = F
         
    # Eliminating Parcels with NaN values -Small Parctrainels-
    X_Clean , k = Cleaning(data_raw,D_x,D_y,F)

    # Shuffling the data set before splitting
    #np.random.shuffle(ain = Scalling(X_train,ND,F)
    
    # Creating the label array 
    Y = Labelling_RNN(X_Clean,D_x,ND,k,F)
    Y_true = Labelling(X_Clean,D_x,ND,k,F)
    Id = IDing(X_Clean,D_x,ND,k,F)
    
    # Filling my Traing data with features from all parcels for all dates
    X = Matrix2Tensor(X_Clean,D_x,D_y,ND,k,B,F)
        
    # Adding labels (classes to each parcel)
        
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, Y_train, Y_test, Y_true_train, Y_true_test, Id_train, Id_test  = train_test_split(X, Y, Y_true, Id,  test_size = test_size , random_state = 0)
    
    # Feature Scaling
    X_train = Scalling(X_train,ND,F)
    X_test  = Scalling(X_test,ND,F)
    
    # Adding dates to the scaled Data
    X_train , X_test = TimeStamping(X_train,len(X_train),dates,ND,F) , TimeStamping(X_test,len(X_test),dates,ND,F)
    
    if train:
        return X_train, Y_train, Y_true_train, Id_train, (input_dim + 5)
    else :
        return X_test , Y_test, Y_true_test, Id_test 
  
    
    
"""   
def main():
    Data_Preprocessing('2016_Optical_25_ID.csv', True)

if __name__ == '__main__':
    main()    
"""  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    