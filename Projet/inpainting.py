# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:04:09 2020

@author: ZixuanFENG & ArnaudDELOL
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.colors as colors
import random
import copy

COLOR_MISS=-100
STEP=10
H_PATCH=12

############################## BASIC FILE OPERATION ##############################

#lire une image et de la renvoyer sous forme d’un numpy.array
def read_im(fn):    
    img=plt.imread(fn)
    img=img/255    

    img_hsv=colors.rgb_to_hsv(img)
    ones=np.ones(img_hsv.shape)
    img_hsv=img_hsv*2-ones
    
    return img_hsv
    
#affichage de l’image et des pixels manquants
def show_im(img):        
    ones=np.ones(img.shape)
    img_rgb=(img+ones)/2
    img_rgb=colors.hsv_to_rgb(img_rgb)    
    
    plt.imshow(img_rgb)
    
#enregistrer l'image dans le repertoire fn(filename)
def save_img(img_hsv,fn):
    ones=np.ones(img_hsv.shape)
    img_rgb=(img_hsv+ones)/2
    img_rgb=colors.hsv_to_rgb(img_rgb)
    img_rgb[img_rgb>1.0]=1.0
    img_rgb[img_rgb<0.0]=0.0
    matplotlib.image.imsave(fn, img_rgb)
    
############################## BASIC PATCH OPERATION ##############################

#retourner le patch centré en(i,j)et de longueur h d’une image im
def get_patch(i,j,h,im):    
    return im[int(i-h/2):int(i+h/2),int(j-h/2):int(j+h/2),:]

###fonctions de conversions entre patchs et vecteurs
#patch->vector
def patch2vector(patch):
    h=len(patch)
    return patch.reshape((h*h*3))
    
#vector->patch
def vector2patch(vector):
    h=int((len(vector)/3)**0.5)    
    res=np.zeros((h,h,3))
    
    for i in range(h):
        for j in range(h):
            res[i,j]=vector[(i*h+j)*3:(i*h+j+1)*3]
    return res

############################## ADD NOISE FUNCTION ##############################

###fonctions pour bruiter l’image originale
#supprimer au hasard un pourcentage de pixel dans l’image
def noise(img,prc):
    length=len(img)
    res=copy.deepcopy(img)
    
    for i in range(length):
        for j in range(length):
            alea=random.random()
            if alea<=prc:
                res[i][j]=np.array([COLOR_MISS,COLOR_MISS,COLOR_MISS])
    return res
    
#supprimer tout un rectangle de l’image
def delete_rect(img,i,j,height,width):
    top=i-height/2
    bottom=i+height/2
    left=j-width/2
    right=j+width/2
    length=len(img)
    res=copy.deepcopy(img)
    
    for i in range(length):
        for j in range(length):
            if i>=top and i<=bottom and j>=left and j<=right:
                res[i][j]=np.array([COLOR_MISS,COLOR_MISS,COLOR_MISS])
    return res

############################## PREPARE TO INPAINTING ##############################

#renvoyer les patchs de l'image qui contiennent des pixels manquants
def get_patch_miss(img):    
    res=[]    
    pos_i=[]
    pos_j=[]
    for i in range(int(H_PATCH/2),len(img),STEP):
        for j in range(int(H_PATCH/2),len(img),STEP):
            patch=get_patch(i,j,H_PATCH,img)
            if np.isin(COLOR_MISS,patch):
                res.append(patch2vector(patch))
                pos_i.append(i)
                pos_j.append(j)
    return np.array(res),pos_i,pos_j

#renvoyer les patchs manquants en ordre decroissant du nb de pixels presents
#return [(pos_i,pos_j) nb_pixel_present] (type=ndarray)
def get_patch_miss_ordered(img):    
    res=[]
    for i in range(int(H_PATCH/2),len(img),STEP):
        for j in range(int(H_PATCH/2),len(img),STEP):
            patch=get_patch(i,j,H_PATCH,img)
            if np.isin(COLOR_MISS,patch):
                ind_empty=patch==COLOR_MISS
                res.append(((i,j),sum(sum(sum(ind_empty)))))
    res.sort(key= lambda res:res[1])    
    return np.array(res)
    
#renvoyer le dictionnaire: les patchs qui ne contiennent aucun pixel manquant.
def get_dict(img):    
    res=[]    
    for i in range(int(H_PATCH/2),len(img),STEP):
        for j in range(int(H_PATCH/2),len(img),STEP):
            patch=get_patch(i,j,H_PATCH,img)
            if not np.isin(COLOR_MISS,patch):
                res.append(patch2vector(patch))
    return np.array(res)

#rend le vecteur de poids sur le dictionnaire qui approxime au mieux le patch  
def get_patch_similar(patch,dic):      
    indices=patch!=COLOR_MISS
    lasso=linear_model.Lasso(alpha=0.001,max_iter=5000)
    lasso.fit(dic[indices],patch[indices])
    return lasso.coef_ , lasso.predict(dic)

#mettre le patch dans l'image a la position (pos_i,pos_j)
def set_patch(pos_i,pos_j,patch,img):
    res=copy.deepcopy(img)
    for i in range(H_PATCH):
        for j in range(H_PATCH):
            res[int(pos_i-H_PATCH/2+i),int(pos_j-H_PATCH/2+j)]=patch[i,j]
    return res

############################## INPAINTING METHODS ##############################

#reparer l'image en remplissant les patchs aleatoirement
#def random_fill_img(img,fn):
def random_fill_img(img):
    result=copy.deepcopy(img)
    patches_miss,pos_i,pos_j=get_patch_miss(result)
    dic=get_dict(result)
    #save_img(result,(fn+"0.png"))
    #cpt=1
    while len(patches_miss)>0:   
        ind=0
        choose=False
        
        while not choose:
            i=random.randint(0,len(patches_miss)-1)
            indices=patches_miss[i]!=COLOR_MISS
            if True in indices:
                choose=True
                ind=i
                break

        w,res=get_patch_similar(patches_miss[ind],dic.T)
        res_patch=vector2patch(res)
        result=set_patch(pos_i[ind],pos_j[ind],res_patch,result)
        patches_miss,pos_i,pos_j=get_patch_miss(result)
        dic=get_dict(result)

        #save_img(result,(fn+str(cpt)+".png"))
        #cpt+=1
    return result

#reparer l'image en remplissant les patchs a l'ordre gauche->droite, haut->bas
# ex. 1 2 3
#    4 5 6
#def simple_fill_img(img,fn):
def simple_fill_img(img):
    result=copy.deepcopy(img)
    patches_miss,pos_i,pos_j=get_patch_miss(result)
    dic=get_dict(result)
    #save_img(result,(fn+"0.png"))
    #cpt=1
    while len(patches_miss)>0:   
        ind=0
        choose=False
        for i in range(len(patches_miss)):
            indices=patches_miss[i]!=COLOR_MISS
            if True in indices:
                choose=True
                ind=i
                break
                
        if choose:
            w,res=get_patch_similar(patches_miss[ind],dic.T)
            res_patch=vector2patch(res)
            result=set_patch(pos_i[ind],pos_j[ind],res_patch,result)
            patches_miss,pos_i,pos_j=get_patch_miss(result)
            dic=get_dict(result)
            
            #save_img(result,(fn+str(cpt)+".png"))
            #cpt+=1
        else:
            return result
    return result
                           
def better_fill_img(img):
    result=copy.deepcopy(img)
    patches_miss_ordered=get_patch_miss_ordered(result)     
    dic=get_dict(result)
                           
    while len(patches_miss_ordered)>0:
        for i in range(len(patches_miss_ordered)):              
            if patches_miss_ordered[i][1]==0:
                break
            else:
                patch=get_patch(patches_miss_ordered[i,0][0],patches_miss_ordered[i,0][1],H_PATCH,result)
                w,res=get_patch_similar(patch2vector(patch),dic.T)
                res_patch=vector2patch(res)
                result=set_patch(patches_miss_ordered[i,0][0],patches_miss_ordered[i,0][1],res_patch,result)

        patches_miss_ordered=get_patch_miss_ordered(result)
        dic=get_dict(result)
                           
    return result