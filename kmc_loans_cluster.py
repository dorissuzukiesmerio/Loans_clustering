import pandas
import numpy
# import random
import matplotlib.pyplot as pyplot

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data = pandas.read_csv('clustering.csv')
print(data.head())
print(data.columns)


#### Simple example with 2 variables:
subset = data[["LoanAmount","ApplicantIncome"]]
subset = subset.values

# # DATA VISUALIZATION:
pyplot.scatter(subset[:,0],subset[:,1])
pyplot.xlabel('AnnualIncome')
pyplot.ylabel('Loan Amount (In Thousands)')
pyplot.savefig("scatter_loanamount_applicantincome.png")
pyplot.close()

def run_kmeans(n, subset):
	machine = KMeans(n_clusters = n) # Constructing the machine
	machine.fit(subset) # fitting the subset
	results = machine.predict(subset) # predicting

	centroids = machine.cluster_centers_
	ssd = machine.inertia_ #sum of square deviations
	
	silhouette = 0
	if n>1:
		silhouette = silhouette_score(subset, machine.labels_, metric = 'euclidean')

	pyplot.scatter(subset[:,0], subset[:,1], c = results) # row, column (all rows, first column)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker= "*", s=200)
	pyplot.savefig("scatterplot_colors_"+ str(n) + ".png") # no need for this if using Jupyter Notebook
	pyplot.close()

	return ssd, silhouette

# result =[]
# for i in range(10):
# 	ssd = run_kmeans(i+1, subset)
# 	result.append(ssd)

result = [run_kmeans(i + 1, subset) for i in range (10)]
print(result)

# result_diff=[]
# for counter, value enumerate(result):
# 	ssd_diff = result[counter-1] - value
# #call counter/index = i , value/element = x

result_diff = [result[i-1] - x for i,x in enumerate(result)][1:]
print(result_diff)

# Visualization of results: 
# Scatterplot (x, y)
# pyplot.scatter(subset[:,0], subset[:,1], c = results) # row, column (all rows, first column)
# pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker= "*", s=200)
# pyplot.savefig("scatterplot_la_colors.png")
# pyplot.close()

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("silhouette.png")
pyplot.close()

print("\nssd: \n", ssd_result)
print("\nssd differences: \n", ssd_result_diff)


print("\nsilhouette scores: \n", silhouette_result)
print("\nmax silhouette scores: \n", max(silhouette_result))
print("\nnumber of cluster with max silhouette scores: \n", silhouette_result.index(max(silhouette_result))+2)
