import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt
from arftools import *
import random


def plot_frontiere_proba(data, f, step=20):
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), 255)


def score(svm, datax, datay):
    return np.mean(svm.predict(datax) == datay)




"""
print("Lineairement séparable")
# Lineairement separable avec un peu de bruit
datax, datay = gen_arti(nbex=1000, data_type=0, epsilon=1)
testx, testy = gen_arti(nbex=1000, data_type=0, epsilon=1)

# lineaire avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='linear')
svm.fit(datax, datay)
plt.figure()
print("Noyau lineaire")
plt.title("Noyau lineaire")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# lineaire avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='linear', C=500)
svm.fit(datax, datay)
plt.figure()
print("Noyau lineaire C élevé")
plt.title("Noyau lineaire C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# poly avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='poly',gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly")
plt.title("Noyau poly")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# poly avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='poly', C=500, degree=5,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly C fort")
plt.title("Noyau poly C fort")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))


# rbf avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='rbf',gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf")
plt.title("Noyau rbf")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# rbf avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='rbf', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf C élevé")
plt.title("Noyau rbf C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))










print("Non lineaire")
# Non-lineairement separable avec un peu de bruit
datax, datay = gen_arti(nbex=1000, data_type=1, epsilon=0.1)
testx, testy = gen_arti(nbex=1000, data_type=1, epsilon=0.1)

# lineaire avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='linear')
svm.fit(datax, datay)
plt.figure()
print("Noyau lineaire")
plt.title("Noyau lineaire")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# lineaire avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='linear', C=500)
svm.fit(datax, datay)
plt.figure()
print("Noyau lineaire C élevé")
plt.title("Noyau lineaire C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# poly avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='poly',gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly")
plt.title("Noyau poly")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# poly avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='poly', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly C fort")
plt.title("Noyau poly C fort")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))


# rbf avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='rbf',gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf")
plt.title("Noyau rbf")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# rbf avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='rbf', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf C élevé")
plt.title("Noyau rbf C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))








print("Echiquier")
# Echiquier separable avec un peu de bruit
datax, datay = gen_arti(nbex=1000, data_type=2, epsilon=0.001)
testx, testy = gen_arti(nbex=1000, data_type=2, epsilon=0.001)

# lineaire avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='linear')
svm.fit(datax, datay)
plt.figure()
print("Noyau lineaire")
plt.title("Noyau lineaire")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# lineaire avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='linear', C=500)
svm.fit(datax, datay)
plt.figure()
print("Noyau lineaire C élevé")
plt.title("Noyau lineaire C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# poly avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='poly',gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly")
plt.title("Noyau poly")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# poly avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='poly', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly C fort")
plt.title("Noyau poly C fort")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))


# rbf avec paramètres par défaut
svm = sklearn.svm.SVC(probability=True, kernel='rbf')
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf")
plt.title("Noyau rbf")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("Parametres par defaut : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

# rbf avec C très fort
svm = sklearn.svm.SVC(probability=True, kernel='rbf', C=500)
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf C élevé")
plt.title("Noyau rbf C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C fort : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

"""

# Grid search
if True : # Mettre à True pour effectuer un grid search
    gammas = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    Cs = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    indices = range(len(datay))
    mscore = 0
    bestg = gammas[0]
    bestc = Cs[0]
    for g in gammas:
        for c in Cs:
            print("Testing g=", g, " c=", c)
            svm = sklearn.svm.SVC(probability=True, kernel='rbf', gamma=g, C=c)
            ind = random.sample(indices, int(len(indices)/len(Cs)))
            svm.fit(np.delete(datax, ind,axis=0), np.delete(datay, ind, axis=0))
            sc = svm.score(datax[ind, :], datay[ind])
            if sc > mscore:
                mscore = sc
                bestg = g
                bestc = c

    print("Meilleur : g=", bestg, " c=", bestc)
    svm = sklearn.svm.SVC(probability=True, kernel='rbf', gamma=bestg, C=bestc)
    svm.fit(datax, datay)
    plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
    plot_data(datax, datay)
    plt.show()
    print("Gaussien VC : ", score(svm, testx, testy))


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


datax, datay = load_usps("USPS_train.txt")
tx, ty = load_usps("USPS_test.txt")

#Multi classes
svms = np.ndarray((10, 10)).astype(sklearn.svm.SVC)
for i in range(9):
    for j in range(i+1, 10):
        svm = sklearn.svm.SVC(kernel='linear')
        svms[i, j] = svm
        indi = datay == i
        indj = datay == j
        ind = np.bitwise_or(indi, indj)
        svm.fit(datax[ind, :], 2 * (datay[ind] == i) - 1)

predicts = np.zeros((len(ty), 10))
for i in range(9):
    for j in range(i+1, 10):
        pred = svms[i, j].predict(tx)
        predicts[:, i] += pred >= 0
        predicts[:, j] += pred < 0

print("Multi classes, One vs one, score : ", np.mean(ty == np.argmax(predicts, axis=1)))
svms = []
for i in range(10):
    svm = sklearn.svm.SVC(kernel='linear')
    svms.append(svm)
    svm.fit(datax, 2*(datay==i)-1)

predicts = np.zeros((len(ty), 10))
for i in range(10):
    predicts[:, i] = svms[i].predict(tx)
print("Multi classes, One vs all, score : ", np.mean(ty == np.argmax(predicts, axis=1)))

#One vs one est plus rapide et plus précis, mais est un peuplus complexe à mettre en place"""