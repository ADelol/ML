from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    loss = np.power(((np.dot(datax , w.T)) - datay), 2)
    return np.sum(loss) / (2 * len(datax))

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    resultat = 0
    taille = len(datax)
    D = datax.shape[1]
    resultat = np.sum(np.dot(datax,w.T)-(datay*datax))/taille
    return resultat


def hinge(datax,datay,w):
    """ retourn la moyenne de l'erreur hinge """
    zeros = np.zeros(datay.shape)
    loss= np.maximum(zeros,np.dot(datax, w.T)*-datay)
    return np.sum(loss) / (2 * len(datax))

def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    tmp = -datay*np.dot(datax, w.T)
    resultat = 0
    taille = len(datax)
    for i in range(taille):
        if tmp[i] <= 0:
            resultat +=  -datax[i]*datay[i]
    return resultat/taille


class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01,biais=False):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.biais = biais

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
        if self.biais:
            D+=1
            b = np.ones((self.N,D))
            b[:,:-1] = datax
            datax = b
            
        self.w = np.random.random((1,D))
        

        for i in range(self.max_iter):
            #lo = self.loss(datax,datay,self.w)
            self.w = self.w - self.eps * self.loss_g(datax,datay,self.w)

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
            
        if datax.shape[1] != self.w.shape[1] and self.biais:
            taille = len(datax)
            D = datax.shape[1]
            D+=1
            b = np.ones((taille,D))
            b[:,:-1] = datax
            datax = b
            

        return (np.dot(datax,self.w.T))

    def score(self,datax,datay):
        taille = len(datax)
        D = datax.shape[1]
        if self.biais:
            D+=1
            b = np.ones((taille,D))
            b[:,:-1] = datax
            datax = b
        
        score = 0
        for i in range(taille):
            print(self.predict(datax[i]))
            print(datay[i])
            if self.predict(datax[i]) == datay[i]:
                score += 1 
        print(score)
        return score/taille



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


def compare_classe(n1,n2,trainx,trainy,testx,testy,loss=mse,loss_g=mse_g,max_iter=1000,eps=0.01,biais=False):
    
    print("Compare deux classes représentant les nombres", n1 ,"et", n2, "en utilisant un perceptron avec la fonction de coût hinge_loss \n")
    
    trainX = trainx[np.where(np.in1d(trainy, [n1,n2]))]
   
    trainY = trainy[np.where(np.in1d(trainy, [n1,n2]))]
    
    testX = testx[np.where(np.in1d(testy, [n1,n2]))]
    testY = testy[np.where(np.in1d(testy, [n1,n2]))]
   
    
    perceptron = Lineaire(loss,loss_g,max_iter=max_iter,eps=eps,biais=biais)
    perceptron.fit(trainX,trainY)
    
    print("Score : train %f, test %f"% (perceptron.score(trainX,trainY),perceptron.score(testX,testY)))
    print(trainX)
    print(trainY)

if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    """    
    plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    #perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.1,biais=True)
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,biais=True)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
"""

    train = load_usps('USPS_train.txt')
    trainx, trainy = train
    test = load_usps('USPS_test.txt')
    testx, testy = test
    compare_classe(6,9,trainx,trainy,testx,testy,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.1,biais=False)
