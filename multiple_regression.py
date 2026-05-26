import numpy as np
import matplotlib.pyplot as plt
# Features:[size, bedrooms, age]
X_train = np.array([
    [850,2,20],
    [900,2,18],
    [1200,3,15],
    [1500,3,10],
    [1700,4,8],
    [2000,4,5],
    [2200,5,4],
    [2500,5,3],
    [2700,5,2],
    [3000,6,1]
])
# House prices
y_train = np.array([
    180,
    200,
    280,
    350,
    420,
    500,
    580,
    670,
    730,
    820
])  

#function for normalization
def normalization(x_train):
    mean=np.mean(x_train,axis=0)
    sd=np.std(x_train,axis=0)
    x_norm=(x_train-mean)/sd
    return x_norm,mean,sd

#function to compute cost
def compute_cost(x_train,y_train,w,b):
    m=x_train.shape[0] #it will return the number of columns we have => no of features we have
    total_cost=0
    for i in range(m):
        prediction=np.dot(x_train[i],w)+b
        error=prediction-y_train[i]
        total_cost+=(error**2)
    total_cost=total_cost/(2*m)
    return total_cost     

def compute_descent(x_train,y_train,w,b):
    m,n=x_train.shape #m=number of training examples #n=number of features
    cost=0
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        prediction=np.dot(w,x_train[i])+b
        error=-y_train[i]+prediction
        for j in range(n):
            dj_dw[j]+=error*x_train[i,j]
        dj_db+=error    
    dj_dw=dj_dw/m
    dj_db=dj_db/m        
    return dj_dw,dj_db
    
def gradient_descent(x_train,y_train,w,b,alpha,iterations):
        cost_history=[]
        for i in range(iterations):
             dj_dw,dj_db=compute_descent(x_train,y_train,w,b)
             w=w-alpha*dj_dw
             b=b-alpha*dj_db
             if i%100==0:
                  cost=compute_cost(x_train,y_train,w,b)
                  print("At iteration:",i,"th"," the cost is:",cost)
                  cost_history.append(cost)
        return w,b,cost_history
      
#intital parameters
x_train,mean,sd=normalization(X_train)
w=np.zeros(x_train.shape[1])
b=0
alpha=0.01
iterations=1000
w_final,b_final,cost_history=gradient_descent(x_train,y_train,w,b,alpha,iterations)
new_house = np.array([3000,4,2])
new_house=(new_house-mean)/sd
prediction=np.dot(w_final,new_house)+b_final
print("Predicted Price:",prediction)
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()