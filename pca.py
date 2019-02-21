# =============================================================================
# title: 			pca.py
# author: 			Nura Kawa
# summary:			Principal component analysis as a sklearn estimator
#					Uses singular value decomposition from numpy.linalg
#					tested with the iris dataset from sklearn
# =============================================================================

# Import Libraries
# -----------------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets
import numpy as np
import math
import numpy.linalg as la
import matplotlib.pyplot as plt

# PCA
# -----------------------------------------------------------------------------
"""
class PCA

	inherits:
				class BaseEstimator from sklearn
				class TransformerMixin from sklearn

	attributes:
				scaled (scale the data?)
				scores (PCA scores)
				components (Matrix of Principal Components)
				loadings (Array of loadings)

	new methods:
				to see all methods run `dir(PCA())`

				__init__ 	follows rules for sklearn estimator
				fit			performs pca on dataset
				_scale		scales data
				plot		plots PCA scores

"""



class PCA(BaseEstimator, TransformerMixin):

	def __init__(self,
	scaled = None,
	scores = None,
	components = None,
	loadings = None):
		self.scaled = scaled
		self.scores = scores
		self.compoments = components
		self.loadings = loadings

	def fit(self, X, scaled = False):

		# assert statements

		assert isinstance(X, np.ndarray), "Please convert dataset to type 'numpy.ndarray'"
		assert scaled in [True, False, None], "`scaled` must be True or False"

		# optional: scale data
		if scaled == True: X = self._scale(X)

		# perform singular value decomposition
		# sometimes SVD does not converge

		try:
			svd_item = la.svd(np.matrix(X, dtype='float'))
		except:
			print("Failed to converge. Try changing your value for `scale`")
			return

		evectors, evalues = svd_item[0], svd_item[1]

		# matrix of principal components
		principal_components = evectors[:,0:len(evalues)]

		for i in range(len(evalues)):
			principal_components[:,i] *= evalues[i]

		# scaled
		self.scaled = scaled
		# components
		self.components = principal_components
		# scores
		self.scores = svd_item[0][:,0:2]
		# loadings
		self.loadings = evalues

		return self

	def _scale(self, df):
		scaled_df = df
		for i in range(df.shape[1]):
			col = df[:,i]
			scaled_col = (col - np.mean(col)) / np.std(col)
			scaled_df[:,i] = scaled_col
		return scaled_df

	def plot(PCA_fitted_object,
	label = None,
	title = None):
		plt.scatter([PCA_fitted_object.scores[:,0]],
		[PCA_fitted_object.scores[:,1]],
		c = label,
		alpha = 0.3)
		plt.title(title)
		plt.xlabel("Component 1")
		plt.ylabel("Component 2")
		plt.show()
		return

# Use with Iris Dataset
# -----------------------------------------------------------------------------

iris = datasets.load_iris()
X = iris.data

# Create and fit a PCA object
iris_pca = PCA().fit(X, scaled = True)
iris_pca.get_params() # this is inherited from class BaseEstimator

# Plot the PCA scores
def color_map(arr):
	map_ = {0:'red',1:'blue',2:'green'}
	colors=arr
	for i in range(len(arr)):
		colors[i] = map_[arr[i]]
	return(colors)

iris_pca.plot(label = color_map(iris.target.tolist()),
title = "PCA View of Iris Dataset")

# resources
# -----------------------------------------------------------------------------
# create an estimator with sklearn
# http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
