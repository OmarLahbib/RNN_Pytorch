import Metrics
import pandas as pd



confusion_matrix=Metrics.ConfusionMatrix(14)


Metrics.ConfusionMatrix.count_predicted(1,1)



v=torch.max(outputs.data, 1)[1]

unique, counts = np.unique(labels_test, return_counts=True)

dict(zip(unique, counts))

for i in range (len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
    
    
    
    
    
    
    
    
#Nuumber of parameters in out model -------------------------------------------    
p=1;  s=0;  
for i in range (len(parameters)):
    p=1
    for j in range (len(parameters[i].size())):
        p = parameters[i].size(j) * p
    s = s + p
        
 

    if  ((epoch - (epoch % 100) ) // 100 ) % 2 == 0:
        old_ratio[epoch%100] = loss.data[0]
    else :
        new_ratio[epoch%100] = loss.data[0]
    
    if (epoch % 200 == 0) and (epoch >200 ) :
        m_new = new_ratio.mean()
        m_old = old_ratio.mean()
        old_ratio = np.zeros((100))
        new_ratio = np.zeros((100))
        
    if (m_old - m_new) / m_old < 0.01 :
        break
    
    
a = pd.DataFrame(2 and 2)    
    
    
prob[:,int(labels[1])]    
    
    
    






unique, counts = np.unique(predictions, return_counts=True)  
dict(zip(unique, counts))

p = predictions

for i in range(0,labels_true_l.shape[0]):
    for j in range (0 , len(counts)):
        if (p[i] == j):
            k = j           
    if ( p[i] != unique[k] ):
        p[i] = unique[k]
    







        













             








































































































    
    
    
    
    
