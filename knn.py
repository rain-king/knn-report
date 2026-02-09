import numpy as np
from statistics import mode

class KNN:
	'''Warning: throws exception, catch with try'''
	def __init__(self,
			  X: np.ndarray, # inference from this var
			  X_train: np.ndarray,
			  y_train: np.ndarray):
		if len(X_train) == len(y_train):
			self.nearest = None
			self.X = X.copy()
			self.X_train = X_train.copy()
			self.y_train = y_train.copy()
		else:
			raise Exception("X debe tener tantos renglones como y")

	'''Returns inference from X with k nearest neighborhoods'''
	def fit(self, k: int) -> np.ndarray:
		y = np.zeros((len(self.X),1), dtype=np.int8)
		if not self.nearest: # checar si existe el cache
			self.nearest = list[list[int]]()
			for i,x in enumerate(self.X):
				distances = np.sum(np.abs(self.X_train - x), axis=1)

				sorted_distances = sorted(
					list(enumerate(distances)),
					key=lambda x: x[1])

				# obtener indice, e insertar a nearest el valor
				# de la clase correspondiente
				self.nearest.append(list[int]())
				for j,_ in sorted_distances:
					self.nearest[i].append(
						int(self.y_train[j][0]))
					######################################
					# !! corrección !!
					# la línea anterior tenía antes
					# sorted_distances[j][0] en vez de j
					# mezclando mal el orden
					######################################

		if self.nearest: # usar cache
			for i in range(len(self.X)):
				y[i] = mode(self.nearest[i][0:k])
		
		return y

