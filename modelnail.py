import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/Data%20of%20nails%20-%20Sheet1.csv")
df['SYMPTOM 1'] = df['SYMPTOM 1'].map({'Crumbling Nail':0.0,'Pitting':1.0,'Change in color,Blood under the nails':2.0,'The nail separates from the bed':3.0,'Nail breaks easily':4.0,'Affects both finger nail and toe nail':5.0,'Drying the nails':6.0,'Typically affects only finger nail':7.0,'Thick Nail':8.0,'Discolored nail that are brown,yellow,white':9.0,'Fragile and cracked nail':10.0,'Discoloration of nail yellow,green or opaque':11.0,'Nail pitting,Nail thickening':12.0,"Bending of nail edges":13.0,'Swelling,Tenderness':14.0,"Redness,soreness":15.0,"Pus":16.0,'Genetics,Injury':17.0,'Circulation issues':18.0,'Ichthyosis':19.0,'Swelling':20.0,'Pain,redness':21.0,'Fever and gland pain':22.0,'Yellow pus':23.0})

df['SYMPTOM 2'] = df['SYMPTOM 2'].map({'Crumbling Nail':0.0,'Pitting':1.0,'Change in color,Blood under the nails':2.0,'The nail separates from the bed':3.0,'Nail breaks easily':4.0,'Affects both finger nail and toe nail':5.0,'Drying the nails':6.0,'Typically affects only finger nail':7.0,'Thick Nail':8.0,'Discolored nail that are brown,yellow,white':9.0,'Fragile and cracked nail':10.0,'Discoloration of nail yellow,green or opaque':11.0,'Nail pitting,Nail thickening':12.0,"Bending of nail edges":13.0,'Swelling,Tenderness':14.0,"Redness,soreness":15.0,"Pus":16.0,'Genetics,Injury':17.0,'Circulation issues':18.0,'Ichthyosis':19.0,'Swelling':20.0,'Pain,redness':21.0,'Fever and gland pain':22.0,'Yellow pus':23.0})

df['SYMPTOM 3'] = df['SYMPTOM 3'].map({'Crumbling Nail':0.0,'Pitting':1.0,'Change in color,Blood under the nails':2.0,'The nail separates from the bed':3.0,'Nail breaks easily':4.0,'Affects both finger nail and toe nail':5.0,'Drying the nails':6.0,'Typically affects only finger nail':7.0,'Thick Nail':8.0,'Discolored nail that are brown,yellow,white':9.0,'Fragile and cracked nail':10.0,'Discoloration of nail yellow,green or opaque':11.0,'Nail pitting,Nail thickening':12.0,"Bending of nail edges":13.0,'Swelling,Tenderness':14.0,"Redness,soreness":15.0,"Pus":16.0,'Genetics,Injury':17.0,'Circulation issues':18.0,'Ichthyosis':19.0,'Swelling':20.0,'Pain,redness':21.0,'Fever and gland pain':22.0,'Yellow pus':23.0})

X = df.drop(['DISEASE','TREATMENT'], axis=1)
df['DISEASE'] = df['DISEASE'].map({'Nail psoriasis':0.0,'Brittle Splitting Nails':1.0,'Nail Fungal Infection':2.0,'Onycholysis':3.0,'Ingrown Toenail':4.0,'Onychogryphosis':5.0,'Paronychia':6.0})
Y = df['DISEASE']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

model1=LogisticRegression()
model1.fit(X,Y)
pickle.dump(model1, open('modelear.pkl','wb'))

model = pickle.load(open('modelear.pkl','rb'))