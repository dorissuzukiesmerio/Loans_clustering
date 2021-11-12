import pandas
import numpy
import random
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans

data = pandas.read_csv('clustering.csv')
print(data.head())
print(data.columns)


#### Simple example with 2 variables:
subset = data[["LoanAmount","ApplicantIncome"]]

# DATA VISUALIZATION:
pyplot.scatter(subset["ApplicantIncome"],subset["LoanAmount"])
pyplot.xlabel('AnnualIncome')
pyplot.ylabel('Loan Amount (In Thousands)')
pyplot.savefig("scatter_loanamount_applicantincome.png")
pyplot.close()


machine = KMeans(n_clusters = 2) # Constructing the machine
machine.fit(subset) # fitting the data
results = machine.predict(subset) # predicting

centroids = machine.cluster_centers_

# Visualization of results: 
# Scatterplot (x, y)
pyplot.scatter(subset[:,0], subset[:,1], c = results) # row, column (all rows, first column)
pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker= "*", s=200)
pyplot.savefig("scatterplot_la_colors.png")
pyplot.close()

