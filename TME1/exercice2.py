# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:06:41 2020

@author: arnaud
"""
from decisiontree import DecisionTree
import pickle
import numpy as np
import matplotlib.pyplot as plt
# data : tableau (films , features ), id2titles : dictionnaire id -> titre ,
# fields : id feature -> nom
[data , id2titles , fields ]= pickle . load ( open ("imdb_extrait.pkl","rb"))

# la derniere colonne est le vote
datax = data [: ,:32]
datay =np. array ([1 if x [33] >6.5 else -1 for x in data ])

dt = DecisionTree ()
dt. max_depth = 2 #on fixe la taille de l’arbre a 5
dt. min_samples_split = 2 # nombre minimum d’exemples pour spliter un noeud
#dt. fit(datax , datay )
#dt. predict ( datax [:5 ,:])
#print (dt. score (datax , datay ))
# dessine l’arbre dans un fichier pdf si pydot est installe .
#dt. to_pdf ("2test_tree.pdf",fields )
# sinon utiliser http :// www. webgraphviz .com/
#dt. to_dot ( fields )
#ou dans la console
#print (dt. print_tree ( fields ))

def x_y(data):
    datax = data [: ,:32]
    datay =np. array ([1 if x [33] >6.5 else -1 for x in data ])
    return datax, datay

#data[row,columns] data[from:to,from:to] data[:] = data[0:len(data)]
def partition(app,data):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    indexLimite = int(app*data.shape[0])
    dataTrain = data[:indexLimite,:]
    dataTest = data[indexLimite:,:]
    return dataTrain,dataTest

def scores_selon_prof(taux_app,data,prof_max):
    scores = []
    x = [i for i in range(2,prof_max)]
    dataTrain, dataTest = partition(taux_app,data)
    train_x, train_y = x_y(dataTrain)
    test_x, test_y = x_y(dataTest)
    
    
    for i in range(2,prof_max):
        dt = DecisionTree()
        dt.max_depth = i
        dt.min_samples_split = 2
        dt.fit(train_x, train_y)
        scores.append(dt. score (test_x , test_y ))
        
    import matplotlib.pyplot as plt
    plt.plot(x,scores)
    plt.ylabel('score en fonction de la profondeur, taux app : ' + str(taux_app))
    plt.savefig(str(taux_app) + "scores.png")
    plt.show()
'''
scores_selon_prof(0.2,data,15)
scores_selon_prof(0.5,data,15)
scores_selon_prof(0.8,data,15)
  '''                
def erreurs(taux_app,data,prof_max):
    erreurs_train = []
    erreurs_test = []
    x = [i for i in range(2,prof_max)]
    dataTrain, dataTest = partition(taux_app,data)
    train_x, train_y = x_y(dataTrain)
    test_x, test_y = x_y(dataTest)
    
    
    for i in range(2,prof_max):
        dt = DecisionTree()
        dt.max_depth = i
        dt.min_samples_split = 2
        dt.fit(train_x, train_y)
        erreurs_train.append(1 - dt.score(train_x , train_y ))
        erreurs_test.append(1 - dt.score(test_x , test_y ))
        
    import matplotlib.pyplot as plt
    plt.plot(x,erreurs_train)
    plt.plot(x,erreurs_test)
    plt.ylabel('erreur en fonction de la profondeur, taux app : ' + str(taux_app))
    plt.legend(['app', 'test'   ], loc='upper left')
    plt.savefig(str(taux_app) + "erreurs.png")
    plt.show()
    

#erreurs(0.2,data,15)
#erreurs(0.5,data,15)
#erreurs(0.8,data,15)

def validation_croisee(n,taux_app,data,prof_max):
    data_app, _ = partition(taux_app,data)
    erreurs_moy_app = []
    borders = np.linspace(0, len(data_app), n + 1, dtype=int)

    for depth in range(1,prof_max):
        print(depth)
        erreurs_test = []
        for i in range(n):
            data_test = data_app[borders[i]:borders[i+1]]
            if len(data_app[0:borders[i]]) > 0:
                data_train = np.concatenate((data_app[0:borders[i]] ,data_app[borders[i+1]:len(data_app)]))
            else:
                data_train = data_app[borders[i + 1]:len(data_app)]

            train_x, train_y = x_y(data_train)
            test_x, test_y = x_y(data_test)

            dt = DecisionTree()
            dt.max_depth = depth
            dt.min_samples_split = 2
            dt.fit(train_x, train_y)
            erreurs_test.append(1 - dt.score(test_x, test_y))
        print(erreurs_moy_app)
        erreurs_moy_app.append((1/n)*np.array(erreurs_test).sum())

    x = [i for i in range(1, prof_max)]
    fig = plt.figure()
    plt.plot(x, erreurs_moy_app)
    plt.xlabel('Erreur moyenne en fonction de la prof avec VC avec taux app de : ' + str(taux_app))
    plt.legend(['app'], loc='upper left')
    plt.savefig(str(taux_app) + "erreursVC.png")
    #plt.show()
    return



validation_croisee(10,0.2,data,13)
validation_croisee(10,0.5,data,13)
validation_croisee(10,0.8,data,13)
validation_croisee(10,1,data,13)