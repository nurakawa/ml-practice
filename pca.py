# =============================================================================
# title: 			pca.py
# author: 			Nura Kawa
# date: 			20 March 2017
# summary:			implements principal component analysis in python
#					from singular value decomposition
# data:				subset of mnist dataset
# =============================================================================

# Import Libraries
import scipy as sp
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

# Set Plot Features
plt.style.use('ggplot')

def main(scaled):

	# read in data
	df = pd.read_csv('../Desktop/spring-2017/r-workshop/mnist_test.csv')
	label = df.iloc[:,0] # save the label
	del df['0'] # deletes the label column - we do not want to perform PCA with it
	
	# optional: scale your data
	if scaled == True:
		df = Scale(df)
	
	# make the components
	svd_item = Princomp(df)
	
	# keep the eigenvalues and eigenvectors
	eigenvalues = svd_item[1]
	eigenvectors = svd_item[0]
	
	# compute the principal components
	columns = []
	for i in range(len(df.columns)):
		columns.append("Component" + str(i+1))
	
	principal_components = pd.DataFrame(index = range(len(df.index)),columns = columns)
		
	for i in range(len(principal_components.columns)):
		principal_components.iloc[:,i] = eigenvalues[i] * eigenvectors[i] 
	
	# plot the PCA scores
	PCA_Scores_Plot(svd_item, label)
	
	# plot the screeplot
	Screeplot(svd_item)
	
	# plot the loadings
	PCA_Plot(principal_components)	
	
def Scale(df):
	# df should be a pandas DataFrame
	for i in range(len(df.columns)):
	#for i in range(ncol):
		col = df.iloc[:,i]
		scaled_col = (col - np.mean(col)) / np.std(col)
		df.iloc[:,i] = scaled_col
	return df

def Princomp(data):
	# data should be pandas dataframe
	mat = pd.DataFrame.as_matrix(data)
	return la.svd(mat)
		
def Screeplot(princomp_item):
	eigs = princomp_item[1] # the singular values
	sorted_eigs = sorted(eigs, reverse = True)
	sorted_eigs = sorted_eigs[0:10] # only need to keep 10 components
	plt.plot(sorted_eigs, marker = 'o')
	plt.xlabel('Number of Components')
	plt.ylabel('Eigenvalues')
	plt.title('Screeplot')
	plt.grid(True)
	plt.show()

def PCA_Plot(component_matrix):
	comp_1 = component_matrix["Component1"]
	comp_2 = component_matrix["Component2"]
	plt.scatter(comp_1, comp_2)
	plt.grid(True)
	plt.title('PCA Loadings')
	plt.xlabel('Component1')
	plt.ylabel('Component2')
	plt.show()

def PCA_Scores_Plot(svd_item, label):
	scores_1 = svd_item[0][:,0]
	scores_2 = svd_item[0][:,1]
	
	plt.scatter(scores_1, scores_2, c = label)
	
	plt.title('PCA Scores')
	plt.xlabel('Component1')
	plt.ylabel('Component2')
	plt.show()

	
if __name__ == "__main__":
    main(False)
	
	
