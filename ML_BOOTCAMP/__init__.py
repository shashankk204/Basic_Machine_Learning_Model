import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalized(x):
  return ((x-x.mean(axis=0))/(x.max(axis=0)-x.min(axis=0)))
#linear regression
class linear_regressor:
  def __init__(self,x,y):
    self.x=normalized(x)
    self.y=y
    self.m,self.n=self.x.shape
    self.w=np.zeros(self.n)
    self.b=0
  def gradient_descent(self,a,iter):
    self.point=np.zeros((2,iter))
    plt.grid(True)
    plt.xlabel("iteration")
    plt.ylabel("J")
    for i in range(iter):
      xdotw_plus_b_y=np.dot(self.x,self.w)+self.b-self.y
      self.b=self.b-((a/self.m)*(np.sum(xdotw_plus_b_y)))
      self.w=self.w-((a/self.m)*(np.dot(self.x.T,xdotw_plus_b_y)))
      self.j=np.sum((np.dot(self.x,self.w)+self.b-self.y)**2)/(2*self.m)
      self.point[0,i]=i+1
      self.point[1,i]=self.j
      if i%50==0:
        print(f"for iterations={i+1} cost function={self.j}")
  def graph(self):
    plt.plot(self.point[0],self.point[1],c="r")
    plt.show()
    plt.clf()
    plt.grid(True)
    plt.xlabel("A")
    plt.ylabel("Label")
    plt.scatter(self.x[:,0],self.y,c="red")
    plt.scatter(self.x[:,0],np.dot(self.x,self.w)+self.b,c="g")
    plt.legend(["train", "test"],loc="upper left")
    plt.show()
    plt.clf()
  def score(self):
    r2=1-(np.sum((np.dot(self.x,self.w)+self.b-self.y)**2)/np.sum((self.y-self.y.mean())**2))
    r2adj=1-((1-r2)*(self.m-1)/(self.m-self.n-1))
    print(r2adj)
    print(self.j)
  def predict(self,x_test):
     x_test=normalized(x_test)
     return np.dot(x_test,self.w)+self.b
"""============================================================"""
#polynomial regression

def poly(x,pow):
  m=x.shape[0]
  n=(((pow)*(pow+1)*((2*pow)+1))+(9*(pow)*(pow+1))+(12*pow))//12
  x_poly=np.zeros((m,n))
  k=0
  i=0
  j=0
  while(pow!=0):
    x_poly[:,k]=((x[:,0]**(i))*(x[:,1]**(j))*(x[:,2]**(pow-i-j)))
    j=j+1
    k=k+1
    if(j>(pow-i)):
      i=i+1
      j=0
    if(i>pow):
      pow=pow-1
      i=0
  return normalized(x_poly)
def polynomial_regression(x,y,a,iter,const=0):
  #setting parameters
  m,n=x.shape
  w=np.zeros(n)
  b=0
  #setting up graph
  point=np.zeros((2,iter))
  plt.grid(True)
  plt.xlabel("iteration")
  plt.ylabel("J") 
  #running gradient descent algorithm
  for i in range(iter):
    xdotw_plus_b_y=np.dot(x,w)+b-y
    b1=b-((a/m)*(np.sum(xdotw_plus_b_y)))
    w1=w-((a/m)*(np.dot(x.T,xdotw_plus_b_y)+(const*w)))
    b=b1
    w=w1
    #calculating different value of j with time
    j=np.sum((np.dot(x,w)+b-y)**2)/(2*m)
    point[0,i]=i+1
    point[1,i]=j
    if i%500==0:
      print(f"for iteration={i} value of J={j}")
        
  #plot graph
  plt.plot(point[0],point[1],c="r")
  plt.show()
  plt.clf()
  #calulating r2 and j
  r2=1-(np.sum((np.dot(x,w)+b-y)**2)/np.sum((y-y.mean())**2))
  r2adj=1-((1-r2)*(m-1)/(m-n-1))
  print("R2=",r2adj)
  print("J=",j)
  return w,b,j
"""=================================================================="""
#logistic regression
def logistic_regression(x,y,a,iter):
  #setting parameters
  m,n=x.shape
  #creating classifiers
  classifiers=np.unique(y)
  y_temp=np.zeros((m,len(classifiers)))
  for i in range(len(classifiers)):
    y_temp[:,i]=y
    if(classifiers[i]==0):
      y_temp[:,i]=y_temp[:,i]+1
      y_temp[:,i][y_temp[:,i]!=1]=0
    else:  
      y_temp[:,i][y_temp[:,i]!=classifiers[i]]=0
      y_temp[:,i][y_temp[:,i]==classifiers[i]]=1
  w=np.zeros((len(classifiers),n))
  b=np.zeros(len(classifiers))  
  #graph
  points=np.zeros((2,iter))

  for i in range(iter):
    z=np.matmul(x,w.T)+b
    f_w_b=1/(1+np.exp(-z))
    b1=b-((a/m)*np.sum((f_w_b-y_temp),axis=0))
    w1=w-((a/m)*(np.matmul((f_w_b-y_temp).T,x)))
    w=w1
    b=b1
    #calculating J
    z=np.matmul(x,w.T)+b
    f_w_b=1/(1+np.exp(-z))
    j=(-1/m)*(np.sum(((1-y_temp)*np.log(1-f_w_b))+(y_temp*np.log(f_w_b))))
    points[0,i]=i+1
    points[1,i]=j
    if i%50==0:
      print(f"for iteration={i+1} value of J={j}")
  plt.grid(True)
  plt.xlabel("iterations")
  plt.ylabel("J")
  plt.plot(points[0,:],points[1,:])
  plt.show()
  print("J=",j)
  return w,b

def accuracy(x,w,b,y):
    z=np.matmul(x,w.T)+b
    f_w_b=1/(1+np.exp(-z))
    q=np.unique(abs(y-np.argmax(f_w_b,axis=1)),return_counts=True)
    print(q[1][0]*100/x.shape[0])
"""===================================================================="""
#KNN
class KNN:
  def __init__(self,x,y):
    self.x=normalized(x)
    self.y=y
    self.m,self.n=x.shape
  def predict(self,x_test):
    x_test=normalized(x_test)
    m_test=x_test.shape[0]
    distance=np.zeros((self.m,2))
    result=np.zeros(m_test)
    k=int(m_test**0.5)
    if k%2==0:
        k=k+1
    for i in range(m_test):
        distance[:,1]=self.y
        distance[:,0]=(((self.x-x_test[i])**2).sum(axis=1))**0.5
        distance=distance[distance[:,0].argsort()]
        ct=np.array(np.unique(distance[0:k,1],return_counts=True)).T
        ct=ct[ct[:,1].argsort()]
        result[i]=ct[-1,0]
    return result
def accuracy_knn(y_pred,y):
    q=np.unique(abs(y-y_pred),return_counts=True)
    print(q[1][0]*100/y.shape[0])

