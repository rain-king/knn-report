import numpy as np
import pandas as pd
from matplotlib import pyplot, image
from os.path import isfile

bands_filenames = "band1.irs band2.irs band3.irs band4.irs".split(" ")

bands = list[np.ndarray]() # 0 corresponds to band1
for i,filename in enumerate(bands_filenames):
	bands.append(np.fromfile(filename, dtype=np.int8))
	bands[i] = bands[i].reshape((512,512))
	if not isfile(f"banda{i+1}.png"):
		image.imsave(f"banda{i+1}.png", bands[i], cmap='gray')
	# else:
	# 	print("Already saved")

# Load training set in pandas to check
colnames = ["band1", "band2", "band3", "band4", "is_water"]
df = pd.read_csv('rsTrain.dat', sep=r'\s+', names=colnames)

# obvious conversion
# df["is_water"] = df["is_water"].astype(np.int8)
df = df.astype(np.int8)
df.dtypes
len(df)

# comenté este codigo a una función para que no tenga que correr
# mientras hago pruebas
def analysis(df):
	# check for possible conversion of all columns to int8
	df_int8 = df.astype(np.int8)
	# no difference, floating point precision not needed
	np.sum(np.abs(df_int8 - df))
	df = df_int8
	# no NA
	df.isna().sum()

	# no blatant outliers
	df.hist(figsize=(6,7))
	pyplot.savefig('hist.png', dpi=150)

# to numpy
X = df.iloc[:,:4].to_numpy(dtype=np.int8)
y = df.iloc[:,4:5].to_numpy(dtype=np.int8)

# train-test split
import random
random.seed(590)

indexes = list(range(200))
random.shuffle(indexes)

train, test = indexes[:160], indexes[160:200]

y[train].mean() # 0.5
y[test].mean() # 0.5

# X[train].mean(axis=0) - X[test].mean(axis=0)
# [0.7     0.38125 0.79375 0.7    ]
# X.std(axis=0)
# [3.7018779  3.64540464 5.24956189 7.87437458]

from sklearn.preprocessing import MinMaxScaler
normal = MinMaxScaler()
X_normal = normal.fit_transform(X)

X_global_normal = (X - X.min()) / (X.max() -  X.min())
X_global_normal.max(axis=0)
