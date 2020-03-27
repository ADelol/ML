# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:22:25 2020

@author: Zixuan FENG
"""
#Exo1.3
import decisiontree
import pickle
import numpy as np

def Exo1_3(data):
    datay=np.array([1 if x[33]>6.5 else -1 for x in data])    
    maxGain=0 #le maximum du gain
    maxInd=-1 #l'indice du meilleur attribut
    
    #pour chaque attribut binaire
    for i in range(28):
        #partition induite: fils0, fils1        
        fils0=[] #fils0: attribut==0
        fils1=[] #fils1: attribut==1
        for j in range(len(data)):
            if data[j][i]==0:
                fils0.append(datay[j])
            else:
                fils1.append(datay[j])

        #calculer l'entropie et l'entropie conditionnelle
        e=decisiontree.entropy(datay)
        ec=decisiontree.entropy_cond([np.array(fils0),np.array(fils1)])
        #print("entropie=",e)
        #print("entropie conditionnelle=",ec)
        
        #calculer le gain d'information
        '''
        gain==0 <=> entropie==entropie_condi <=> le test sur cet attribut sert a rien
        gain==1 <=> on peut 100% classifier l'exemple selon la valeur de cet attribut
        '''
        gain=e-ec
        #print("gain ",fields[i]," :",gain)        
        
        #trouver le meilleur attribut
        if gain>maxGain:
            maxGain=gain
            maxInd=i
        
    return maxInd
        
        

#data:tableau(films,features), id2titles:dictionnaire id->titre,
#fields:id feature->nom
[data,id2titles,fields]=pickle.load(open("imdb_extrait.pkl","rb"))
#la derniere colonne est le vote
datax=data[:,:32]
#datay=np.array([1 if x[33]>6.5 else -1 for x in data])
print("L'indice du meilleur attribut: ",Exo1_3(data))


















