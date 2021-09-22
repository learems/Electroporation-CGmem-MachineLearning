# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 05:07:53 2021

@author: Xinru & Fangwei
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('D:/File/MD/MLAnalysis')
from class_MLAnalysis_7_22 import MLAnalysis

PATH = 'D:/File/MD/MLAnalysis/DataTables/'

#Load APM Data
for i in range(1,5):    
    locals()['apor'+str(i)] = pd.read_csv (PATH+'APM2x2_mem'+str(i)+'_porated.csv')
    locals()['anon'+str(i)] = pd.read_csv (PATH+'APM2x2_mem'+str(i)+'_nonporated.csv')
X_APM = pd.concat([apor1,apor2,apor3,apor4,anon1,anon2,anon3,anon4],ignore_index=True)    
X_APM.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM = np.concatenate([np.ones(len(apor1)+len(apor2)+len(apor3)+len(apor4)),np.zeros(len(anon1)+len(anon2)+len(anon3)+len(anon4))],axis=0) 
Y_APM = Y_APM.astype('int')

#Load BPM data
for i in range(1,5):    
    locals()['bpor'+str(i)] = pd.read_csv (PATH+'BPM2x2_mem'+str(i)+'_porated.csv')
    locals()['bnon'+str(i)] = pd.read_csv (PATH+'BPM2x2_mem'+str(i)+'_nonporated.csv')
X_BPM = pd.concat([bpor1,bpor2,bpor3,bpor4,bnon1,bnon2,bnon3,bnon4],ignore_index=True)
X_BPM.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM = np.concatenate([np.ones(len(bpor1)+len(bpor2)+len(bpor3)+len(bpor4)),np.zeros(len(bnon1)+len(bnon2)+len(bnon3)+len(bnon4))],axis=0) 
Y_BPM = Y_BPM.astype('int')

#Load APM-hyp Data
for i in range(1,5):    
    locals()['apor'+str(i)+'_hyp'] = pd.read_csv (PATH+'APM2x2-hyp_mem'+str(i)+'_porated.csv')
    locals()['anon'+str(i)+'_hyp'] = pd.read_csv (PATH+'APM2x2-hyp_mem'+str(i)+'_nonporated.csv')
X_APM_hyp = pd.concat([apor1_hyp,apor2_hyp,apor3_hyp,apor4_hyp,anon1_hyp,anon2_hyp,anon3_hyp,anon4_hyp],ignore_index=True)    
X_APM_hyp.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_hyp = np.concatenate([np.ones(len(apor1_hyp)+len(apor2_hyp)+len(apor3_hyp)+len(apor4_hyp)),np.zeros(len(anon1_hyp)+len(anon2_hyp)+len(anon3_hyp)+len(anon4_hyp))],axis=0) 
Y_APM_hyp = Y_APM_hyp.astype('int')

#Load BPM-hyp Data
for i in range(1,5):    
    locals()['bpor'+str(i)+'_hyp'] = pd.read_csv (PATH+'BPM2x2-hyp_mem'+str(i)+'_porated.csv')
    locals()['bnon'+str(i)+'_hyp'] = pd.read_csv (PATH+'BPM2x2-hyp_mem'+str(i)+'_nonporated.csv')
X_BPM_hyp = pd.concat([bpor1_hyp,bpor2_hyp,bpor3_hyp,bpor4_hyp,bnon1_hyp,bnon2_hyp,bnon3_hyp,bnon4_hyp],ignore_index=True)    
X_BPM_hyp.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM_hyp = np.concatenate([np.ones(len(bpor1_hyp)+len(bpor2_hyp)+len(bpor3_hyp)+len(bpor4_hyp)),np.zeros(len(bnon1_hyp)+len(bnon2_hyp)+len(bnon3_hyp)+len(bnon4_hyp))],axis=0) 
Y_BPM_hyp = Y_BPM_hyp.astype('int')

#Load APM-40us Data
for i in range(1,3):    
    locals()['apor'+str(i)+'_40us'] = pd.read_csv (PATH+'APM2x2-40us_mem'+str(i)+'_porated.csv')
    locals()['anon'+str(i)+'_40us'] = pd.read_csv (PATH+'APM2x2-40us_mem'+str(i)+'_nonporated.csv')
X_APM_40us = pd.concat([apor1_40us,apor2_40us,anon1_40us,anon2_40us],ignore_index=True)    
X_APM_40us.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_40us = np.concatenate([np.ones(len(apor1_40us)+len(apor2_40us)),np.zeros(len(anon1_40us)+len(anon2_40us))],axis=0) 
Y_APM_40us = Y_APM_40us.astype('int')


#APM predict APM-hyp
MDA = MLAnalysis.PrepareData(X_APM,Y_APM,X_APM_hyp,Y_APM_hyp,pattern='ab')
MDA.ML(model='RF')

#APM mix BPM predict APM mix BPM 
X_mix = pd.concat([X_APM,X_BPM],ignore_index=True)
Y_mix = np.concatenate([Y_APM,Y_BPM],axis=0) 
X_mix_hyp = pd.concat([X_APM_hyp,X_BPM_hyp],ignore_index=True)
Y_mix_hyp = np.concatenate([Y_APM_hyp,Y_BPM_hyp],axis=0) 
MDA = MLAnalysis.PrepareData(X_mix,Y_mix,X_mix_hyp,Y_mix_hyp,pattern='ab')
MDA.ML(model='RF')


#Create data with only five the most important features
X_APM_PUGMFSCHOLPC = X_APM[['PU','GM','FS','CHOL','PC']]
X_APM_hyp_PUGMFSCHOLPC = X_APM_hyp[['PU','GM','FS','CHOL','PC']]
X_BPM_PUGMFSCHOLPC = X_BPM[['PU','GM','FS','CHOL','PC']]
X_BPM_hyp_PUGMFSCHOLPC = X_BPM_hyp[['PU','GM','FS','CHOL','PC']]
X_APM_40us_PUGMFSCHOLPC = X_APM_40us[['PU','GM','FS','CHOL','PC']]

MDA = MLAnalysis.PrepareData(X_APM_PUGMFSCHOLPC,Y_APM,X_BPM_PUGMFSCHOLPC,Y_BPM,pattern='aa')
MDA.ML(model='RF')


#two mem predict the other two
X_APM_mem12 = pd.concat([apor1,apor2,anon1,anon2],ignore_index=True)    
X_APM_mem12.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_mem12 = np.concatenate([np.ones(len(apor1)+len(apor2)),np.zeros(len(anon1)+len(anon2))],axis=0) 
Y_APM_mem12 = Y_APM_mem12.astype('int')

X_APM_mem34 = pd.concat([apor3,apor4,anon3,anon4],ignore_index=True)    
X_APM_mem34.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_mem34 = np.concatenate([np.ones(len(apor3)+len(apor4)),np.zeros(len(anon3)+len(anon4))],axis=0) 
Y_APM_mem34 = Y_APM_mem34.astype('int')

X_APM_mem12_PUGMFSCHOLPC = X_APM_mem12[['PU','GM','FS','CHOL','PC']]
X_APM_mem34_PUGMFSCHOLPC = X_APM_mem34[['PU','GM','FS','CHOL','PC']]

MDA = MLAnalysis.PrepareData(X_APM_mem12_PUGMFSCHOLPC,Y_APM_mem12,X_APM_mem34_PUGMFSCHOLPC,Y_APM_mem34,pattern='ab')
MDA.ML(model='RF')

MDA = MLAnalysis.PrepareData(X_APM_mem12,Y_APM_mem12,X_APM_mem34,Y_APM_mem34,pattern='ab')
MDA.ML(model='RF')

X_APM_mem13 = pd.concat([apor1,apor3,anon1,anon3],ignore_index=True)    
X_APM_mem13.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_mem13 = np.concatenate([np.ones(len(apor1)+len(apor3)),np.zeros(len(anon1)+len(anon3))],axis=0) 
Y_APM_mem13 = Y_APM_mem13.astype('int')

X_APM_mem24 = pd.concat([apor2,apor4,anon2,anon4],ignore_index=True)    
X_APM_mem24.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_mem24 = np.concatenate([np.ones(len(apor2)+len(apor4)),np.zeros(len(anon2)+len(anon4))],axis=0) 
Y_APM_mem24 = Y_APM_mem24.astype('int')

MDA = MLAnalysis.PrepareData(X_APM_mem13,Y_APM_mem13,X_APM_mem24,Y_APM_mem24,pattern='ab')
MDA.ML(model='RF')

X_APM_mem13_PUGMFSCHOLPC = X_APM_mem13[['PU','GM','FS','CHOL','PC']]
X_APM_mem24_PUGMFSCHOLPC = X_APM_mem24[['PU','GM','FS','CHOL','PC']]

MDA = MLAnalysis.PrepareData(X_APM_mem13_PUGMFSCHOLPC,Y_APM_mem13,X_APM_mem24_PUGMFSCHOLPC,Y_APM_mem24,pattern='ab')
MDA.ML(model='RF')


X_APM_mem14 = pd.concat([apor1,apor4,anon1,anon4],ignore_index=True)    
X_APM_mem14.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_mem14 = np.concatenate([np.ones(len(apor1)+len(apor4)),np.zeros(len(anon1)+len(anon4))],axis=0) 
Y_APM_mem14 = Y_APM_mem14.astype('int')

X_APM_mem23 = pd.concat([apor2,apor3,anon2,anon3],ignore_index=True)    
X_APM_mem23.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_APM_mem23 = np.concatenate([np.ones(len(apor2)+len(apor3)),np.zeros(len(anon2)+len(anon3))],axis=0) 
Y_APM_mem23 = Y_APM_mem23.astype('int')

MDA = MLAnalysis.PrepareData(X_APM_mem14,Y_APM_mem14,X_APM_mem23,Y_APM_mem23,pattern='ab')
MDA.ML(model='RF')

X_APM_mem14_PUGMFSCHOLPC = X_APM_mem14[['PU','GM','FS','CHOL','PC']]
X_APM_mem23_PUGMFSCHOLPC = X_APM_mem23[['PU','GM','FS','CHOL','PC']]

MDA = MLAnalysis.PrepareData(X_APM_mem14_PUGMFSCHOLPC,Y_APM_mem14,X_APM_mem23_PUGMFSCHOLPC,Y_APM_mem23,pattern='ab')
MDA.ML(model='RF')


X_BPM_mem12 = pd.concat([bpor1,bpor2,bnon1,bnon2],ignore_index=True)    
X_BPM_mem12.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM_mem12 = np.concatenate([np.ones(len(bpor1)+len(bpor2)),np.zeros(len(bnon1)+len(bnon2))],axis=0) 
Y_BPM_mem12 = Y_BPM_mem12.astype('int')

X_BPM_mem34 = pd.concat([bpor3,bpor4,bnon3,bnon4],ignore_index=True)    
X_BPM_mem34.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM_mem34 = np.concatenate([np.ones(len(bpor3)+len(bpor4)),np.zeros(len(bnon3)+len(bnon4))],axis=0) 
Y_BPM_mem34 = Y_BPM_mem34.astype('int')

MDA = MLAnalysis.PrepareData(X_BPM_mem12,Y_BPM_mem12,X_BPM_mem34,Y_BPM_mem34,pattern='ab')
MDA.ML(model='SVM')

X_BPM_mem12_PUGMFSCHOLPC = X_BPM_mem12[['PU','GM','FS','CHOL','PC']]
X_BPM_mem34_PUGMFSCHOLPC = X_BPM_mem34[['PU','GM','FS','CHOL','PC']]

MDA = MLAnalysis.PrepareData(X_BPM_mem12_PUGMFSCHOLPC,Y_BPM_mem12,X_BPM_mem34_PUGMFSCHOLPC,Y_BPM_mem34,pattern='ab')
MDA.ML(model='RF')



X_BPM_mem13 = pd.concat([bpor1,bpor3,bnon1,bnon3],ignore_index=True)    
X_BPM_mem13.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM_mem13 = np.concatenate([np.ones(len(bpor1)+len(bpor3)),np.zeros(len(bnon1)+len(bnon3))],axis=0) 
Y_BPM_mem13 = Y_BPM_mem13.astype('int')

X_BPM_mem24 = pd.concat([bpor2,bpor4,bnon2,bnon4],ignore_index=True)    
X_BPM_mem24.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM_mem24 = np.concatenate([np.ones(len(bpor2)+len(bpor4)),np.zeros(len(bnon2)+len(bnon4))],axis=0) 
Y_BPM_mem24 = Y_BPM_mem24.astype('int')

MDA = MLAnalysis.PrepareData(X_BPM_mem13,Y_BPM_mem13,X_BPM_mem24,Y_BPM_mem24,pattern='ab')
MDA.ML(model='SVM')

X_BPM_mem13_PUGMFSCHOLPC = X_BPM_mem13[['PU','GM','FS','CHOL','PC']]
X_BPM_mem24_PUGMFSCHOLPC = X_BPM_mem24[['PU','GM','FS','CHOL','PC']]

MDA = MLAnalysis.PrepareData(X_BPM_mem13_PUGMFSCHOLPC,Y_BPM_mem13,X_BPM_mem24_PUGMFSCHOLPC,Y_BPM_mem24,pattern='ab')
MDA.ML(model='RF')


X_BPM_mem14 = pd.concat([bpor1,bpor4,bnon1,bnon4],ignore_index=True)    
X_BPM_mem14.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM_mem14 = np.concatenate([np.ones(len(bpor1)+len(bpor4)),np.zeros(len(bnon1)+len(bnon4))],axis=0) 
Y_BPM_mem14 = Y_BPM_mem14.astype('int')

X_BPM_mem23 = pd.concat([bpor2,bpor3,bnon2,bnon3],ignore_index=True)    
X_BPM_mem23.drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=True)
Y_BPM_mem23 = np.concatenate([np.ones(len(bpor2)+len(bpor3)),np.zeros(len(bnon2)+len(bnon3))],axis=0) 
Y_BPM_mem23 = Y_BPM_mem23.astype('int')

MDA = MLAnalysis.PrepareData(X_APM_mem14,Y_APM_mem14,X_APM_mem23,Y_APM_mem23,pattern='ab')
MDA.ML(model='RF')

X_BPM_mem14_PUGMFSCHOLPC = X_BPM_mem14[['PU','GM','FS','CHOL','PC']]
X_BPM_mem23_PUGMFSCHOLPC = X_BPM_mem23[['PU','GM','FS','CHOL','PC']]

MDA = MLAnalysis.PrepareData(X_BPM_mem14_PUGMFSCHOLPC,Y_BPM_mem14,X_BPM_mem23_PUGMFSCHOLPC,Y_BPM_mem23,pattern='ab')
MDA.ML(model='RF')

MDA = MLAnalysis.PrepareData(X_BPM_hyp,Y_BPM_hyp,X_APM,Y_APM,pattern='ab')
MDA.ML(model='RF')



#Feature importance plot
Columns = [X_APM.columns[i] for i in range(len(X_APM.columns))]
Columns[0] = 'APL'
Columns[1] = 'thickness'
Columns[2] = 'mean curvature'
Columns[3] = ''r'cos $\theta_{dip}$'
Columns[4] = 'charge'
Columns[5] = 'tail order P2'

sns.barplot(x=MDA.model.feature_importances_, y=Columns)
plt.xlabel('Feature Importance Score',fontsize=18)
plt.ylabel('Features',fontsize=18)
plt.title("BPM-hyp",fontsize=20)
plt.show()



#Threshold
#Train model
MDA = MLAnalysis.PrepareData(X_BPM,Y_BPM,X_BPM_hyp,Y_BPM_hyp,pattern='ab')
MDA.ML(model='RF')

#Overall false positive rate
#Defined as the ratio of predict 1 but actual is 0 and actual is 0
fp = 0
tn = 0
for i in range(len(MDA.prediction_test)):
    if ((MDA.Y_test[i] == 0) and (MDA.prediction_test[i] == 1)):
      fp = fp+1
    if (MDA.Y_test[i] == 0):
      tn = tn+1
fpr = fp/tn
print('overall false positive rate = ',fpr)

#Overall false negative rate
#Defined as the ratio of predict 0 but actual is 1 and actual is 1 
fn = 0
tp = 0
for i in range(len(MDA.prediction_test)):
    if ((MDA.Y_test[i] == 1) and (MDA.prediction_test[i] == 0)):
      fn = fn+1
    if (MDA.Y_test[i] == 1):
      tp = tp+1
fnr = fn/tp
print('overall false negative rate = ',fnr)

#Define Function
def CountNum(X):
    X_du = X.drop_duplicates()
    count = np.zeros(len(X_du))
    for j in range(len(X)):  
        for i in range(len(X_du)):
            if (X.iloc[j] == X_du.iloc[i]).all():
                count[i] = count[i]+1
                break
    return count

def Calculate(mem,pore_loc):
    true_posotive = 0
    for i in range(len(mem)):
        for j in range(len(pore_loc)):
            if (mem.iloc[i] == pore_loc.iloc[j]).all():
                true_posotive = true_posotive+1
    #Calculate false_posotive_rate
    #Define as the ratio of actual is 1 but predict 0 and actual is 1 (missing pore)            
    false_negative_rate = (len(pore_loc) - true_posotive)/len(pore_loc)
    
    #Calculate false_negative_rate
    #Define as the ratio of actual is 0 but predict 1 and actual is 1 (wrong pore)
    false_posotive_rate = (len(mem) - true_posotive)/len(mem)
    
    return false_negative_rate,false_posotive_rate


#Load real Data
for i in range(1,5):    
    locals()['pore'+str(i)] = pd.read_csv (PATH+'BPM2x2-hyp_mem'+str(i)+'_porated.csv')
    locals()['non'+str(i)] = pd.read_csv (PATH+'BPM2x2-hyp_mem'+str(i)+'_nonporated.csv')
    locals()['pore'+str(i)+'_loc'] = locals()['pore'+str(i)][['x_rel','y_rel']].drop_duplicates()
    locals()['X_mem'+str(i)] = pd.concat([locals()['pore'+str(i)],locals()['non'+str(i)]],ignore_index=True)
    #locals()['X_mem'+str(i)+'_drop'] = locals()['X_mem'+str(i)][['PU','GM','FS','CHOL','PC']]
    locals()['predict_mem'+str(i)] = MDA.model.predict(locals()['X_mem'+str(i)].drop(['x_rel','y_rel','tpore','frame'],axis=1,inplace=False))
    #locals()['Bool_mem'+str(i)] = [False if locals()['predict_mem'+str(i)][j]==0 else True for j in range(len(locals()['predict_mem'+str(i)]))]

Bool_mem1 = [False if predict_mem1[i]==0 else True for i in range(len(predict_mem1))]
Bool_mem2 = [False if predict_mem2[i]==0 else True for i in range(len(predict_mem2))]
Bool_mem3 = [False if predict_mem3[i]==0 else True for i in range(len(predict_mem3))]
Bool_mem4 = [False if predict_mem4[i]==0 else True for i in range(len(predict_mem4))]

mem1 = X_mem1.iloc[Bool_mem1][['x_rel','y_rel']]
mem2 = X_mem2.iloc[Bool_mem2][['x_rel','y_rel']]
mem3 = X_mem3.iloc[Bool_mem3][['x_rel','y_rel']]
mem4 = X_mem4.iloc[Bool_mem4][['x_rel','y_rel']]

for i in range(1,5):    
    locals()['count'+str(i)] = CountNum(locals()['mem'+str(i)])   
    locals()['mem'+str(i)].drop_duplicates(inplace=True)

threshold = 90
Bool1_mem1 = [True if count1[i]>threshold else False for i in range(len(mem1))]
mem1_predict = mem1.iloc[Bool1_mem1]
Bool2_mem2 = [True if count2[i]>threshold else False for i in range(len(mem2))]
mem2_predict = mem2.iloc[Bool2_mem2]
Bool3_mem3 = [True if count3[i]>threshold else False for i in range(len(mem3))]
mem3_predict = mem3.iloc[Bool3_mem3]
Bool4_mem4 = [True if count4[i]>threshold else False for i in range(len(mem4))]
mem4_predict = mem4.iloc[Bool4_mem4]


for i in range(1,5):    
    locals()['false_negative_rate_mem'+str(i)], locals()['false_posotive_rate_mem'+str(i)]= Calculate(locals()['mem'+str(i)+'_predict'],locals()['pore'+str(i)+'_loc'])
    print(locals()['false_negative_rate_mem'+str(i)],locals()['false_posotive_rate_mem'+str(i)])    

#Draw plots 
real = []
real.append(pore1_loc)  
real.append(pore2_loc) 
real.append(pore3_loc)  
real.append(pore4_loc)  

pre = []
pre.append(mem1_predict)
pre.append(mem2_predict)
pre.append(mem3_predict)
pre.append(mem4_predict)

plt.style.use('ggplot')
fig,ax=plt.subplots(2,2,figsize=(9,9))
axx=ax.flatten()
plt.subplots_adjust(wspace = 0.25, hspace = 0.45)          
for a in enumerate(axx):          
    a[1].scatter(real[a[0]]['x_rel'],real[a[0]]['y_rel'],s=25,marker='o',c='cyan')
    a[1].scatter(pre[a[0]]['x_rel'],pre[a[0]]['y_rel'],s=25,c='red',marker='.')     
    a[1].set_title("mem " + str(a[0]+1),fontsize=18,pad=15)      
    a[1].tick_params(axis="y", labelsize=14)      
    a[1].tick_params(axis="x", labelsize=14)
plt.suptitle('BPM predict APM-hyp with threshold 90',fontsize=20)
plt.show()
        


