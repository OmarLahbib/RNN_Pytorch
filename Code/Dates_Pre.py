import datetime as dt
import calendar
from datetime import datetime
import numpy as np

#-------Convert text file containing dates to Season Month Week Day Numpy array
#------------------------------------------------------------------------------
def season(var):
    if var in [1,2] or var == 12 :
        return 0
    elif var in [3,5] :
        return 1
    elif var in [6,8] :
        return 2
    else :
        return 3
    
#------------------------------------------------------------------------------    
def month(var):
    if var % 3 == 0   :
        return 0
    elif var % 3 == 1 :
        return 1
    else :
        return 2
    
#------------------------------------------------------------------------------    
def week(year, month, day):
    calendar.setfirstweekday(0)
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0]
    return(week_of_month)
    
#------------------------------------------------------------------------------    
def days_between(d1,m1,y1,d2,m2,y2):
    day_1 = dt.datetime(y1,m1,d1)
    day_2 = dt.datetime(y2,m2,d2)
    return abs((day_2 - day_1).days)
   
#------------------------------------------------------------------------------    
def Dates(file_name):
    #file_name = 'Dates.txt'
    dates = np.loadtxt(file_name, dtype=int , delimiter=',')
    
    ND=dates.shape[0]
    Dates_Numpy=np.zeros((ND,5))
    
    Y0= dates[0]//10000
    M0=(dates[0]//100)%100
    D0=dates[0]%100
    
    Dates_Numpy[0,0] = season(M0)
    Dates_Numpy[0,1] = month(M0)
    Dates_Numpy[0,2] = week(Y0,M0,D0)
    Dates_Numpy[0,3] = dt.datetime( Y0 , M0 , D0 ).weekday()
    Dates_Numpy[0,4] = 0
    
    for i in range(1,ND):
        
        Y= dates[i]//10000
        M=(dates[i]//100)%100
        D=dates[i]%100
        
        Dates_Numpy[i,0] = season(M)
        Dates_Numpy[i,1] = month(M)
        Dates_Numpy[i,2] = week(Y,M,D)
        Dates_Numpy[i,3] = dt.datetime( Y , M , D ).weekday()
        Dates_Numpy[i,4] = days_between(D,M,Y,D0,M0,Y0)
        
        Y0= dates[i]//10000
        M0=(dates[i]//100)%100
        D0=dates[i]%100
    
    for i in range(0,ND):
        Dates_Numpy[i,0] = ( Dates_Numpy[i,0] - Dates_Numpy[:,0].min())/(Dates_Numpy[:,0].max()-Dates_Numpy[:,0].min())
        Dates_Numpy[i,1] = ( Dates_Numpy[i,1] - Dates_Numpy[:,1].min())/(Dates_Numpy[:,1].max()-Dates_Numpy[:,1].min())
        Dates_Numpy[i,2] = ( Dates_Numpy[i,2] - Dates_Numpy[:,2].min())/(Dates_Numpy[:,2].max()-Dates_Numpy[:,2].min())
        Dates_Numpy[i,3] = ( Dates_Numpy[i,3] - Dates_Numpy[:,3].min())/(Dates_Numpy[:,3].max()-Dates_Numpy[:,3].min())
        Dates_Numpy[i,4] = ( Dates_Numpy[i,4] - Dates_Numpy[:,4].min())/(Dates_Numpy[:,4].max()-Dates_Numpy[:,4].min())
    
    return Dates_Numpy , ND














