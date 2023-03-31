import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#neural_network

def normalized(x):
  return ((x-x.mean(axis=0))/(x.max(axis=0)-x.min(axis=0)))
def activation_function(activation,x):
  if activation=="ReLu":
    x[x<0]=x[x<0]*0.01
    return x

  elif activation=="softmax":
    ex=np.exp(x)
    return ((ex.T)/(np.sum(ex,axis=1))).T
  
def activation_derivative(activation,x):
  if activation=="ReLu":
     k=np.array((x>0),dtype=int)
     k[k<0]=0.01
     return k

  elif activation=="softmax":
    return (activation_function("softmax",x))*(1-activation_function("softmax",x))
class dense:
  def __init__(self,units,n):
    self.units=units
    self.n=n
    np.random.seed(10)
    self.w=np.random.randn(units,n)
    self.b=np.zeros(units)
    self.a=0
    self.dj_dz=0
class neural_network:
  network=np.array([])
  no_layers=0

  def __init__(self,x,y):
    self.x=normalized(x)
    self.y=y
    self.m,self.n=x.shape
    self.network=np.append(self.network,dense(0,self.n))
    self.network[0].a=x
    self.no_layers=self.no_layers+1

  def add_layer(self,units):
   self.network=np.append(self.network,dense(units[0],(self.network[self.no_layers-1].n)))
   self.network[self.no_layers].w=self.network[self.no_layers].w*np.sqrt(2/(self.network[self.no_layers-1].n))
   self.no_layers=self.no_layers+1 
   for i in range(1,len(units)):
     self.network=np.append(self.network,dense(units[i],(self.network[self.no_layers-1].units)))
     self.network[self.no_layers].w=self.network[self.no_layers].w*np.sqrt(2/(self.network[self.no_layers-1].n))
     self.no_layers=self.no_layers+1
   self.y_one_hot_end=np.zeros((self.m,units[-1]))
   self.network[self.no_layers-1].w=self.network[self.no_layers-1].w*np.sqrt(1/2)
   for i in range(self.m):
    self.y_one_hot_end[i,self.y[i]]=1
  
  def forward_prop(self,X,k):
    layer=self.network[self.no_layers-k+1]
    w=layer.w
    b=layer.b
    z=normalized(np.matmul(X,w.T)+b)
    if(k==2):
      layer.a=(activation_function("softmax",z))
      layer.dj_dz=layer.a-self.y_one_hot_end
    else:
      layer.a=normalized(activation_function("ReLu",z))
      self.forward_prop(layer.a,k-1)
  def back_prop(self,alfa,iter):
    self.point=np.zeros((2,iter))
    for i in range(iter):
      k=self.no_layers
      while(k>=2):
        self.forward_prop(self.x,self.no_layers)
        layer=self.network[k-1]
        prev_layer=self.network[k-2]
        if k<(self.no_layers):
          dj_dz=np.matmul(self.network[k].dj_dz,self.network[k].w)*(activation_derivative("ReLu",layer.a))
          layer.dj_dz=dj_dz
      
        dj_dw=np.matmul(layer.dj_dz.T,prev_layer.a)/self.m
        dj_db=np.sum(layer.dj_dz,axis=0)/(self.m)
        layer.w=layer.w-(alfa*dj_dw)
        layer.b=layer.b-(alfa*dj_db)
        k=k-1
      j=np.sum((self.y_one_hot_end*np.log(self.network[self.no_layers-1].a)))/(-1*self.m)
      self.point[0,i]=i+1
      self.point[1,i]=j
      if i%10==0:
            print(j ,"=== at iter=",i)   
      if j>=self.point[1,i-1]:
        alfa=alfa/10
  def graph(self):
    plt.grid(True)
    plt.xlabel("iteration")
    plt.ylabel("J")
    plt.plot(self.point[0],self.point[1],c="r")
    plt.show()
    plt.clf()
  def predict(self,x_test):
    x_test=normalized(x_test)
    self.forward_prop(x_test,self.no_layers)
    return np.argmax(self.network[-1].a,axis=1)
def accuracy_nn(y_pred,y):
    q=np.unique(abs(y-y_pred),return_counts=True)
    print(q[1][0]*100/y.shape[0])