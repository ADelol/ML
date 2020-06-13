from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    print(datay.shape)
    print(w.T.shape)
    print(datax.shape)
    print(np.dot(datax , w.T).shape)
    print(w.T)
    loss = np.power(((np.dot(datax , w.T)) - datay), 2)
    return np.sum(loss) / (2 * len(datax))

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    print((np.dot(datax.T,np.dot(datax , w.T) - datay) / len(datax)).shape)
    return np.snp.dot(datax.T,np.dot(datax , w.T) - datay) / len(datax)

def hinge(datax,datay,w):
    """ retourn la moyenne de l'erreur hinge """
    zeros = np.zeros(datay.shape)
    loss= np.maximum(zeros,np.dot(datax, w.T)*-datay)
    return np.sum(loss) / (2 * len(datax))

def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    tmp = np.dot(datax, w.T)*datay
    res = 0
    for i in range(len(tmp)):
        if tmp[i] <= 0:
            res = res + datax[i]*datay[i]
    return res
            
    pass

class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        self.N = len(datay)
        datax = datax.reshape(self.N,-1)
        D = datax.shape[1]
        self.w = np.random.random((D,))
        print("a")
        print(self.w)
        for i in range(self.max_iter):
            #lo = self.loss(datax,datay,self.w)
            self.w = self.w - self.eps * self.loss_g(datax,datay,self.w)

    def predict(self,datax):
        datax = datax.reshape(self.N,-1)
        D = datax.shape[1]
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        return np.dot(datax,self.w.T)

    def score(self,datax,datay):
        datax = datax.reshape(self.N,-1)
        D = datax.shape[1]
        return self.loss(datax,datay,self.w)



def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()



if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """

    plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)

 