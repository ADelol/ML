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

plt.ion()
trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
plt.figure()
perceptron = sklearn.linear_model.Perceptron(max_iter=5 , tol=1e-3)
perceptron.fit(trainx,trainy)
print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
plt.figure()
plt.title("Perceptron sklearn")
plot_frontiere(trainx,perceptron.predict,200)
plot_data(trainx,trainy)
"""


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
print("C élevé : ", score(svm, testx, testy))
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

# poly avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='poly', C=500, degree=5,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly C élevé")
plt.title("Noyau poly C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C élevé : ", score(svm, testx, testy))
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

# rbf avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='rbf', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf C élevé")
plt.title("Noyau rbf C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C élevé : ", score(svm, testx, testy))
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
print("C élevé : ", score(svm, testx, testy))
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

# poly avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='poly', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly C élevé")
plt.title("Noyau poly C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C élevé : ", score(svm, testx, testy))
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

# rbf avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='rbf', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau rbf C élevé")
plt.title("Noyau rbf C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C élevé : ", score(svm, testx, testy))
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
print("C élevé : ", score(svm, testx, testy))
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

# poly avec C très élevé
svm = sklearn.svm.SVC(probability=True, kernel='poly', C=500,gamma='scale')
svm.fit(datax, datay)
plt.figure()
print("Noyau poly C élevé")
plt.title("Noyau poly C élevé")
plot_frontiere_proba(datax, lambda x: svm.predict_proba(x)[:, 0], step=50)
plot_data(datax, datay)
plt.show()
print("C élevé : ", score(svm, testx, testy))
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
print("C elevé : ", score(svm, testx, testy))
print("Nombre de vecteurs supports : " + str(len(svm.support_vectors_)))

"""
"""
# Grid search
if False : # Mettre à True pour effectuer un grid search
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
"""



trainx, trainy = load_usps("USPS_train.txt")
testx, testy = load_usps("USPS_test.txt")


classifieurs = [[sklearn.svm.SVC(kernel='linear') for j in range(10)] for i in range(9)]

for i in range(9):
    for j in range(i+1, 10):
        svm = classifieurs[i][j]
        indi = trainy == i
        indj = trainy == j
        ind = np.bitwise_or(indi, indj)
        print(ind)
        svm.fit(trainx[ind, :], 2 * (trainy[ind] == i) - 1)

predictions = np.zeros((len(testy), 10))
for i in range(9):
    for j in range(i+1, 10):
        pred = classifieurs[i][j].predict(testx)
        predictions[:, i] += pred >= 0
        predictions[:, j] += pred < 0
print(predictions.shape)

print("One-vs-one : ", np.mean(testy == np.argmax(predictions, axis=1)))



classifieurs = [sklearn.svm.SVC(kernel='linear').fit(trainx, 2*(trainy==i)-1) for i in range(10)]
predicts = np.zeros((len(testy), 10))
for i in range(10):
    predicts[:, i] = classifieurs[i].predict(testx)
print("One-versus-all : ", np.mean(testy == np.argmax(predicts, axis=1)))