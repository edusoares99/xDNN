import math
import numpy as np
from scipy.spatial.distance import cdist
import copy
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

def xDNN(Input,Mode):
    if Mode == 'Learning':
        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
        RBPARAM = PrototypesIdentification(Images,Features,Labels,CN)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = RBPARAM
        MemberLabels = {}
        for i in range(0,CN+1):
           MemberLabels[i]=Input['Labels'][Input['Labels']==i] 
        Output['xDNNParms']['CurrentNumberofClass']=CN+1
        Output['xDNNParms']['OriginalNumberofClass']=CN+1
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output

    elif Mode == 'Updating':
        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = Input['xDNNParms']['OriginalNumberofClass']
        RBPARAM = Input['xDNNParms']['Parameters']
        RBPARAM = PrototypesUpdating(Images,Features,Labels,CN,RBPARAM)
        Output = {}
        Output['xDNNParms'] = {}        
        Output['xDNNParms']['Parameters'] = RBPARAM
        MemberLabels=Input['xDNNParms']['MemberLabels']
        for i in range(1,CN+1):
           MemberLabels[i]=[MemberLabels(i),Input['Labels'][Input['Labels']==i]]
        Output['xDNNParms']['CurrentNumberofClass']=CN
        Output['xDNNParms']['OriginalNumberofClass']=CN
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output
    
    elif Mode == 'Validation':
        Params=Input['xDNNParms']
        datates=Input['Features']
        Test_Results = TestResult(Params,datates)
        EstimatedLabels = Test_Results['EstimatedLabels'] 
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        return Output
    

def PrototypesIdentification(Image,GlobalFeature,LABEL,CL):
    data = {}
    image = {}
    label = {}
    RBPARAM = {}
    for i in range(0,CL+1):
        seq = np.argwhere(LABEL==i)
        data[i]=GlobalFeature[seq,]
        image[i] = {}
        for j in range(0, len(seq)):
            image[i][j] = Image[seq[j][0]]
        label[i] = np.ones((len(seq),1))*i
    for i in range(0, CL+1):
        RBPARAM[i] = xDNNclassifier(data[i],image[i])
    return RBPARAM
        

def xDNNclassifier(Data,Image):
    L, N, W = np.shape(Data)
    radius = 1 - math.cos(math.pi/6)
    data = Data.copy()
    Centre = data[0,]
    Center_power = np.power(Centre,2)
    X = np.sum(Center_power)
    Support =np.array([1])
    Noc = 1
    GMean = Centre.copy()
    Radius = np.array([radius])
    ND = 1
    VisualPrototype = {}
    VisualPrototype[1] = Image[0]
    for i in range(2,L+1):
        GMean = (i-1)/i*GMean+data[i-1,]/i
        CentreDensity=np.sum((Centre-np.kron(np.ones((Noc,1)),GMean))**2,axis=1)
        CDmax=max(CentreDensity)
        CDmin=min(CentreDensity)
        DataDensity=np.sum((data[i-1,] - GMean) ** 2)
        if i == 2:
            distance = cdist(data[i-1,].reshape(1,-1),Centre.reshape(1,-1),'euclidean')[0]
        else:
            distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
        value,position= distance.max(0),distance.argmax(0)
        value=value**2
        
        if DataDensity > CDmax or DataDensity < CDmin or value > 2*Radius[position]:
            Centre=np.vstack((Centre,data[i-1,]))
            Noc=Noc+1
            VisualPrototype[Noc]=Image[i-1]
            X=np.vstack((X,ND))
            Support=np.vstack((Support, 1))
            Radius=np.vstack((Radius, radius))
        else:
            Centre[position,] = Centre[position,]*(Support[position]/Support[position]+1)+data[i-1]/(Support[position]+1)
            Support[position]=Support[position]+1
            Radius[position]=0.5*Radius[position]+0.5*(X[position,]-sum(Centre[position,]**2))/2  
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centre
    dic['Support'] =  Support
    dic['Radius'] =  Radius
    dic['GMean'] =  GMean
    dic['Prototype'] = VisualPrototype
    dic['L'] =  L
    dic['X'] =  X
    return dic  
 
    
def PrototypesUpdating(Image,GlobalFeature,LABEL,CL,RBPARAM):
    data = {}
    image = {}
    label = {}
    RBPARAM = {}
    for i in range(1,CL+1):
        seq = np.argwhere(LABEL==i)
        data[i]=GlobalFeature[seq,]
        image[i] = {}
        for j in range(0, len(seq)):
            image[i][j] = Image[seq[j][0]]
        label[i] = np.ones((len(seq),1))*i
    for i in range(1, CL+1):
        RBPARAM[i] = xDNNclassifier_onlineupdating(data[i],image[i],RBPARAM[i])
    return RBPARAM
    

def TestResult(Params,datates):
    PARAM=Params['Parameters']
    CurrentNC=Params['CurrentNumberofClass']
    LAB=Params['MemberLabels']
    VV = 1
    LTes=np.shape(datates)[0]
    EstimatedLabels = np.zeros((LTes))
    Scores=np.zeros((LTes,CurrentNC))
    for i in range(1,LTes + 1):
        data = datates[i-1,]
        R=np.zeros((VV,CurrentNC))
        Value=np.zeros((CurrentNC,1))
        for k in range(0,CurrentNC):
            distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'minkowski',6))[0]
            Value[k]=distance[0]
        Value = np.exp(-1*Value**2).T
        Scores[i-1,] = Value
        Value = Value[0]
        Value_new = np.sort(Value)[::-1]
        indx = np.argsort(Value)[::-1]
        EstimatedLabels[i-1]=indx[0]
    LABEL1=np.zeros((CurrentNC,1))
    
    
    for i in range(0,CurrentNC): 
        LABEL1[i] = np.unique(LAB[i])
    #    a,c=np.histogram(LAB[i]+1, np.unique(LAB[i]+1)[0])
    #    b = np.unique(LAB[i])
    #    t = np.argmax(a)
    #    LABEL1[i-1] = b[t]
    #    print(LABEL1)
    EstimatedLabels = EstimatedLabels.astype(int)
    EstimatedLabels = LABEL1[EstimatedLabels]   
    dic = {}
    dic['EstimatedLabels'] = EstimatedLabels
    dic['Scores'] = Scores
    return dic
         

def xDNNclassifier_onlineupdating(Data,Image,PARAM):
    L, W = np.shape(Data)
    radius = 0.00001
    Xnorm = np.sqrt(np.sum(Data**2,axis=1))
    Xnorm = Xnorm.reshape(-1, 1)
    data = Data / Xnorm[:,(np.ones((1,W),int)-1)[0]]
    ND = 1
    Noc = PARAM['Noc']
    Centre=PARAM['Centre']
    Support=PARAM['Support']
    Radius=PARAM['Radius']
    GMean=PARAM['GMean']
    VisualPrototype=PARAM['Prototype']
    K=PARAM['L']
    X=PARAM['X']
    for i in range(K+1,L+K+1):
        GMean = (i-1)/i*GMean+data[i-1-K,]/i
        CentreDensity=np.sum((Centre-np.kron(np.ones((Noc,1)),GMean))**2,axis=1)
        CDmax=max(CentreDensity)
        CDmin=min(CentreDensity)
        DataDensity=np.sum((data[i-1-K,] - GMean) ** 2)
        if i == 2:
            distance = cdist(data[i-1-K,].reshape(1,-1),Centre.reshape(1,-1),'euclidean')[0]
        else:
            distance = cdist(data[i-1-K,].reshape(1,-1),Centre,'euclidean')[0]
        value,position= distance.max(0),distance.argmax(0)
        value=value**2
        
        if DataDensity > CDmax or DataDensity < CDmin or value > 2*Radius[position]:
            Centre=np.vstack((Centre,data[i-1-K,]))
            Noc=Noc+1
            VisualPrototype[Noc]=Image[i-K-1]
            X=np.vstack((X,ND))
            Support=np.vstack((Support, 1))
            Radius=np.vstack((Radius, radius))
        else:
            Centre[position,] = Centre[position,]*(Support[position]/Support[position]+1)+data[i-1-K]/(Support[position]+1)
            Support[position]=Support[position]+1
            Radius[position]=0.5*Radius[position]+0.5*(X[position,]-sum(Centre[position,]**2))/2   
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centre
    dic['Support'] =  Support
    dic['Radius'] =  Radius
    dic['GMean'] =  GMean
    dic['Prototype'] = VisualPrototype
    dic['L'] =  L+K
    dic['X'] =  X
    return dic    
    


from numpy import genfromtxt
# Load the files, including features, images and labels. /Users/eduardosoares/Downloads/xDNN/data_df_X_test.csv

X_train_file_path = r'/Users/eduardosoares/Downloads/xDNN/data_df_X_train.csv'

y_train_file_path = r'/Users/eduardosoares/Downloads/xDNN/data_df_y_train.csv'

X_test_file_path = r'/Users/eduardosoares/Downloads/xDNN/data_df_X_test.csv'

y_test_file_path = r'/Users/eduardosoares/Downloads/xDNN/data_df_y_test.csv'

X_train = genfromtxt(X_train_file_path, delimiter=',')

y_train = pd.read_csv(y_train_file_path,header=None)

X_test = genfromtxt(X_test_file_path, delimiter=',')

y_test = pd.read_csv(y_test_file_path,header=None)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

pd_y_train_labels = y_train[1]
pd_y_train_images = y_train[0]

pd_y_test_labels = y_test[1]
pd_y_test_images = y_test[0]

#print(pd_y_test_labels)
#print(pd_y_test_images)
#print(y_train_labels.dtypes)
#print(y_train_images.dtypes)


y_train_labels = pd_y_train_labels.to_numpy()
y_train_images = pd_y_train_images.to_numpy()

y_test_labels = pd_y_test_labels.to_numpy()
y_test_images = pd_y_test_images.to_numpy()

#print(y_train_labels)
#print()
#print(y_train_images)

Input1 = {}

Input1['Images'] = y_train_images

Input1['Features'] = X_train

Input1['Labels'] = y_train_labels

Mode1 = 'Learning'


Output1 = xDNN(Input1,Mode1)


Input2 = {}

Input2['xDNNParms'] = Output1['xDNNParms']

Input2['Images'] = y_test_images 
Input2['Features'] = X_test
Input2['Labels'] = y_test_labels 
Mode2 = 'Validation' 
Output2 = xDNN(Input2,Mode2)

precision = Output2['ConfMa'][0,0] / Output2['ConfMa'][0,0] + Output2['ConfMa'][1,0]
recall = Output2['ConfMa'][0,0] / Output2['ConfMa'][0,0] + Output2['ConfMa'][0,1]
F1 = (2*precision*recall) / (precision + recall)


print(Output2['ConfMa'])


'''
img = cv2.imread(r'D:\Lancaster_courses\Dissertstion\Data\iROADSDataset\Daylight\Daylight_00000.jpeg')
cv2.imshow('src',img)
#cv2.waitKey(0)
#print(img.shape)
print(img)
Data = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
value1 = xDNNclassifier_online(Data,img)   
#value2 = xDNNclassifier_onlineupdating(Data,value1)
print(value1)
print(value2)       
     
radius = 1 - math.cos(math.pi/6)    
Data = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
L, W = np.shape(Data)
data = Data.copy()
Centre = data[0,]
Center_power = np.power(Centre,2)

X = Center_power.sum(axis = 0)
Support = np.array([1])
GMean = Centre.copy()
NoC = 1
Radius = np.array([radius])
ND=1
#print("Radius is",Radius[0])
#print(GMean) 
#print(data[1,])

for i in range(2,L+1):
    #print(i)
    GMean = (i-1)/i*GMean+data[i-1,]/i
    CentreDensity=np.sum((Centre-np.kron(np.ones((NoC,1)),GMean))**2,axis=1)
    #print(CentreDensity)
    CDmax=max(CentreDensity)
    CDmin=min(CentreDensity)
    DataDensity=np.sum((data[i-1,] - GMean) ** 2)
    print(data[i-1,].reshape(1,-1).shape)
    print(Centre.shape)
    if i == 2:
        distance = cdist(data[i-1,].reshape(1,-1),Centre.reshape(1,-1),'euclidean')[0]
    else:
        distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
    print(distance)
    value,position= distance.min(0),distance.argmin(0)
    value=value**2

    if DataDensity > CDmax or DataDensity < CDmin or value > 2*Radius[position]:
        Centre=np.vstack((Centre,data[i-1,]))
        NoC=NoC+1
        X=np.vstack((X,ND))
        Support=np.vstack((Support, 1))
        Radius=np.vstack((Radius, radius))
        print("RADIUS" , Radius)
        print("Support is" ,Support)
        print("X is : " ,X)
        print(Centre)
        print(Centre.shape)
        print(data[i-1,])
        print("---------")
        #print(value)
        #print(position)
    else:
        Centre[position,] = Centre[position,]*(Support[position]/Support[position]+1)+data[i-1]/(Support[position]+1)
        Support[position]=Support[position]+1
        Radius[position]=0.5*Radius[position]+0.5*(X[position,]-sum(Centre[position,]**2))/2    
    return 
'''