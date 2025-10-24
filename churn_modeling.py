import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from kneed import KneeLocator
from scipy.stats import mode
df=pd.read_csv('Churn_Modelling.csv')
new_df=df.drop(['RowNumber','CustomerId','Surname','Geography','Gender','Age'],axis=1)
Y=new_df['Exited']
X=new_df.drop(['Exited'],axis=1)

class Churn:
    
    def get_best_cluster(X,Y):
        X_train, X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.2,random_state=50)
        intertias=[]
        for i in range(1,11):
            model=KMeans(n_clusters=i)
            model.fit(X_train)
            intertias.append(model.inertia_)
    

        knee=KneeLocator(range(1,11),intertias,curve='convex',direction='decreasing')
        # print(knee.elbow)


        #plotting

        # Uncomment these lines if you want a pictorial representation
        # plt.plot(range(1,11),intertias)
        # plt.xlabel('Number of clusters')
        # plt.ylabel('Intertia')
        # plt.title('Elbow Curve')
        # plt.show()
        return knee.elbow
    
    def get_model(X,Y):
        X_train, X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.2,random_state=50)
        elb_pt=Churn.get_best_cluster(X,Y)
        model=KMeans(n_clusters=elb_pt,random_state=50)
        model.fit(X_train)
        y_pred=model.predict(X_test)
        labels=np.zeros_like(y_pred)
        for i in range(elb_pt):
            mask=(y_pred==i)
            labels[mask]=mode(Y_test[mask],keepdims=True)[0]
        accuracy=accuracy_score(Y_test,labels)
        return accuracy



