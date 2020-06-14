from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    return np.mean((np.dot(datax, w.T) - datay) ** 2)

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    return -np.mean(np.dot(datax,w.T)-(datay*datax))



def hinge(datax,datay,w):
    """ retourn la moyenne de l'erreur hinge """
    zeros = np.zeros(datay.shape)
    loss= np.maximum(zeros,np.dot(datax, w.T)*-datay)
    return np.sum(loss) / (2 * len(datax))

def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    return np.mean(np.dot(datax.T, datay*(datay * np.dot(datax, w.T) < 0)).T, axis=0)


class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01,biais=False):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.wtrajec = []
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
            self.wtrajec.append(self.w)
            self.w = self.w + self.eps * self.loss_g(datax,datay,self.w)

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
            
        return np.dot(datax, self.w.T)

    def score(self,datax,datay):
        taille = len(datax)
        D = datax.shape[1]
        if self.biais:
            D+=1
            b = np.ones((taille,D))
            b[:,:-1] = datax
            datax = b
            
        return np.mean((self.predict(datax).T * datay > 0))



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
     #Partie 1
    plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)

    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.0001,biais=False)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    

    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.0001,biais=False)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)


    #Partie 2
    trainx, trainy = load_usps("USPS_train.txt")
    testx, testy = load_usps("USPS_test.txt")
    perceptron = Lineaire(hinge, hinge_g, 1000, 0.0001,False)
    indices69 = (trainy == 6) + (trainy == 9)
    datay69 = trainy[indices69]
    perceptron.fit(trainx[indices69, :], np.ones(len(datay69))-2*(datay69 == 9))
    print("6 vs 9")
    print(perceptron.w)
    
    
    indices18 = (trainy == 1) + (trainy == 8)
    datay18 = trainy[indices18]
    perceptron.fit(trainx[indices18, :], np.ones(len(datay18))-2*(datay18 ==8))
    print("1 vs 8")
    print(perceptron.w)
    
    train_datay = np.ones(len(trainy)) - 2 * (trainy != 6)
    perceptron.fit(trainx, train_datay)
    print("6 contre les autres")
    print(perceptron.w)

    print("6 contre les autres : courbes d'erreurs")
    step = 100
    iters = range(1, 1001, step)
    sc_train = np.zeros(len(iters))
    sc_test = np.zeros(len(iters))
    test_datay = np.ones(len(ty)) - 2*(ty != 6)
    ioff = 0
    perceptron = Lineaire(hinge, hinge_g, step, 0.00001)
    for iteration in iters:
        perceptron.fit(trainx, train_datay)
        sc_train[ioff] = perceptron.score(trainx, train_datay)
        sc_test[ioff] = perceptron.score(testx, test_datay)
        ioff+=1
    plt.figure()
    plt.plot(iters, sc_train)
    plt.plot(iters, sc_test)
    plt.legend(["Train", "Test"])
    plt.show()
    
    # Partie 3
    # melange de 4 gaussiennes
    trainx,trainy =  gen_arti(nbex=1000,data_type=1,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=1,epsilon=1)
    
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.0001,biais=False)
    perceptron.fit(trainx,trainy)
    print("Erreur pour type 1 : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    
    #Projection 
    
    trainx = np.hstack((trainx, np.reshape((trainx[:, 0]*trainx[:, 1]), (np.shape(trainx)[0], 1)))) #Ajout de x1*x2
    perceptron = Lineaire(hinge, hinge_g, 500, 0.0001)
    perceptron.fit(trainx, trainy)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pos = trainy >= 0
    neg = trainy < 0
    ax.scatter(trainx[pos, 0], trainx[pos, 1], trainx[pos, 2], c="g", marker="x")
    ax.scatter(trainx[neg, 0], trainx[neg, 1], trainx[neg, 2], c="r", marker="o")
    grid, xx,yy = make_grid(xmax=np.max(trainx[:,0]),  xmin=np.min(trainx[:,0]), ymax=np.max(trainx[:,1]), ymin=np.min(trainx[:,1]), step=10)
    weights = perceptron.w[0]
    z = (-weights[0]*xx - weights[1]*yy)*1.0/weights[2]
    surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.3)
    plt.show()
    print("type 1 avec projection : ")
    print(perceptron.score(trainx, trainy))
    
    # echiqieur
    trainx,trainy =  gen_arti(nbex=1000,data_type=2,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=2,epsilon=1)
    
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.0001,biais=False)
    perceptron.fit(trainx,trainy)
    print("Erreur pour type 2 : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    
