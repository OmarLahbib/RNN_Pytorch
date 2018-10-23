import numpy as np
   
def filling_by_batch(labels, outputs, output_dim):
    cm = np.zeros((output_dim,output_dim))

    for i in range(0,int(labels.size(0))):
        cm[ int(labels[i]) , int(outputs[i]) ] += 1
        
    return cm


def MIOU(cm, output_dim): 
    FS = np.zeros((output_dim))
    Pre = np.zeros((output_dim))
    Rec = np.zeros((output_dim))
    
    for i in range(output_dim):
        FS[i]  = cm[i,i] / (  cm[:,i].sum() + cm[i,:].sum() - cm[i,i] ) *100
        Pre[i] = cm[i,i] / (  cm[i,:].sum() ) *100
        Rec[i] = cm[i,i] / (  cm[:,i].sum() ) *100
    
    return  FS, Pre, Rec, FS.mean()


def transform(data_1, data_2, data_3):
    
    for i in range(0,data_1.shape[0]):
        for j in range (0,data_2.shape[0]):
            if ( data_1[i] == j ):
                k = j
                break
            
    data_1[i] = int(data_3[k])
    
    return data_1
