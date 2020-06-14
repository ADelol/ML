import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math

plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)

###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]
grid[:, 0], grid[:, 1] = grid[:, 1], grid[:, 0].copy()

def centers_from_grid(grid):
    '''

    :param grid:
    :return: Renvoie les centres des carres de la grid
    '''
    steps = int(np.sqrt(len(grid)))
    print(steps)
    centers = []
    for i in range(len(grid)-steps):
        if (i + 1)%steps != 0:
            centers.append(np.average((grid[i],grid[i+steps+1]),axis=0))
    return np.array(centers)

class modele_histo():

    def __init__(self):
        self.densite = 0

    def make_histo(self,steps,data,xmin,xmax,ymin,ymax):
        res = np.zeros((steps,steps))
        distance_x = (xmax-xmin)/steps
        distance_y = (ymax - ymin)/steps

        for xi in data:
            x = math.floor((xi[1]-xmin)/distance_x)
            y = math.floor((xi[0] - ymin)/distance_y)
            res[x][y] += 1
        res /=  len(data)
        self.densite = res

        return res
    # renvoie la probabilite que le point x,y soit un poi
    def predict(self,x,y,xmin,xmax,ymin,ymax):
        steps = len(self.densite)
        distance_x = (xmax - xmin) / steps
        distance_y = (ymax - ymin) / steps
        x = math.floor((x - xmin) / distance_x)
        y = math.floor((y - ymin) / distance_y)
        return self.densite[x][y]

class modele_noyau_parzen():

    def __init__(self):
        self.densite = 0

    def fit(self,data,h,grid):
        d = data.shape[1]
        print(len(data))
        d = 2
        densites = []
        for xi in grid:
            dedans = np.sum(np.sum(np.abs(data - xi)/h,axis=1) <= h/d)
            # Ici on ne prend pas le (carre car dim = 2 ) "carre unitaire" mais le carre de longueur h
            # p(x) = k/N*V
            densite_xi = dedans / ((len(data) * h**d))
            densites.append(densite_xi)
        print(str(np.sum(np.array(densites) >= 1)))
        print(densites)

        return np.array(densites)/len(data)

    # renvoie la densite au point donne
    def predict(self,point,data,h):
        d= 2
        dedans = np.sum(np.sum(np.abs(data - point) / h, axis=1) <= h / 2)
        densite_xi = dedans / len(data) / h**d
        return densite_xi


steps = 100
parzen = modele_noyau_parzen()
#il faut faire la grille pour visualiser
res = parzen.fit(geo_mat,0.15,grid).reshape(steps,steps)
plt.figure()
plt.title("parzen")
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()

class modele_noyau_gaussien():

    def __init__(self):
        self.densite = 0

    def fit(self,data,h,grid):
        d = data.shape[1]
        d= 2
        densites = []
        for xi in grid:
            x = np.sum((np.abs(data - xi) / h),axis=1)
            dedans = np.sum( (1/(np.sqrt(2*np.pi))) *np.exp(-0.5*x**2))
            # Suffit de modifier dedans car c'est le noyau donc la façon de selectionner les points dans lhypercube qui change selon le noyau
            densite_xi = dedans / (len(data) * h**d)
            densites.append(densite_xi)
        print(densites)
        print(str(np.sum(np.array(densites) >= 1)))
        return np.array(densites)/len(data)

    # renvoie la densite au point donné
    def predict(self, point, data, h):
        d = 2
        dedans = np.sum((1/2*np.pi)*np.exp(-0.5*(np.sum(np.abs(data - xi)/h,axis=1))**2) <= h/2)
        densite_xi = dedans / (len(data) / h ** d)
        return densite_xi

gaussien = modele_noyau_gaussien()
res = gaussien.fit(geo_mat,0.005,grid).reshape(steps,steps)
plt.figure()
plt.title("gaussian")
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()

''' 
Corriger la legende qui semble pas être correcte'''
###################################################



'''
# A remplacer par res = monModele.predict(grid).reshape(steps,steps)
histo = modele_histo()
res = histo.make_histo(3,geo_mat,xmin,xmax,ymin,ymax)
res = np.matrix.transpose(res)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()

histo = modele_histo()
res = histo.make_histo(3,geo_mat,xmin,xmax,ymin,ymax)
res = np.matrix.transpose(res)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()

histo = modele_histo()
res = histo.make_histo(100,geo_mat,xmin,xmax,ymin,ymax)
res = np.matrix.transpose(res)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
'''

''' prédire la note d’un POI en
fonction de son emplacement. Dans ce dernier contexte de classification supervisée, vous pouvez mettre
en oeuvre l’estimateur de Nadaraya-Watson et les k-plus proches voisins. utiliser lda pour avoir les hashtags clés
implementation de gensim

point wise mutual information'''