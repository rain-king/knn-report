import numpy as np
import pandas as pd
from matplotlib import pyplot, image
import os

# import de modulos propios
from datasets import X_global_normal, X, X_normal, y
from datasets import normal, bands, train, test
from knn import KNN

def inference(x, train_X, train_y, k_list, prefix):
	knn = KNN(x, train_X, train_y)
	for k in k_list:
		inference = knn.fit(k).reshape((512,512))
		image.imsave(f"{prefix}inferencia_k_{k}.png",
			inference, cmap='gray')

try:
	for dataset,nombre in zip((X,X_normal,X_global_normal),
			("Sin transformar",
			"Normalizar por columna",
			"Normalizar todo el dataset")):
		knn_train = KNN(dataset[train], dataset[train], y[train])

		knn = KNN(dataset[test], dataset[train], y[train])
		print(f"Resultado con transformación: {nombre}")
		print("k\t\tTraining Accuracy\tAccuracy\tMCC")
		for k in [3,7,100,150]:
			y_prediction = knn.fit(k)
			y_train_prediction = knn_train.fit(k)
			training_accuracy = np.mean(
				y_train_prediction == y[train])
			accuracy = np.mean(y_prediction == y[test])
		
			fp = np.sum(
				y_prediction[y[test] == 0] == 1
			)
			tp = np.sum(
				y_prediction[y[test] == 1] == 1
			)
			fn = np.sum(
				y_prediction[y[test] == 1] == 0
			)
			tn = np.sum(
				y_prediction[y[test] == 0] == 0
			)

			precision = tp / (tp + fp) if tp + fp > 0 else 1
			recall = tp / (tp + fn) if tp + fn > 0 else 1

			f1_score = 2/(1/precision + 1/recall)\
				if precision > 0 and recall > 0 else 0

			true_positive_rate = tp / (tp + fn)
			true_negative_rate = tn / (tn + fp)
			balanced_accuracy =\
				(true_positive_rate + true_negative_rate)/2

			numerator = (tp*tn - fp*fn)
			denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
			mcc = numerator/denominator if denominator > 0 else 0

			print(f"{k}\t\t{training_accuracy}\t\t\t{accuracy}\t\t{mcc}")


	# image reconstruction
	x = np.zeros((4,512*512))
	for i,band in enumerate(bands):
		x[i] = band.reshape((512*512,))

	x = x.T
	x_global_norm = (x - X.min())/(X.max() - X.min())
	x_norm = normal.transform(x)

	for directory in ("original","per_char","global"):
		os.makedirs(directory, exist_ok=True)

	k_list = [3,7,100,150]
	if True: # cambiar a False para solo generar el análisis estadístico
		inference(x, X, y, k_list, "original/")
		inference(x_norm, X_normal, y, k_list, "per_char/")
		inference(x_global_norm, X_global_normal, y, k_list, "global/")
	
except Exception as e:
	print(e)
