import numpy as np
import matplotlib.pyplot as plt
import math
 
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.cm as cm
import itertools
import seaborn as sns


num_wolves = 15
num_iteration = 300
path1 = './Data_PythonSimulations/socialLearning/python_300_'
path2 = '_4_0.5_15/'
pathList = [(path1 + str(val) + path2) for val in range(1,14)]

distanceList = []
convexHullList = []
centroidList = []

for path in pathList:
	dataList = []
	#Distance from the target Plot
	for i in range(num_wolves):
	     data = np.genfromtxt(path + 'wolf' + str(i)+'.csv', delimiter = ',')
	     dataList.append(data)
	targetDistanceData = np.zeros([num_iteration,2])                
	for wolf in dataList:
	     for i in range(num_iteration):
	                             targetDistanceData[i,0] = wolf[i,0]
	                             targetDistanceData[i,1] += math.log(np.linalg.norm(np.array([593,593]) - wolf[i,1:]))/len(dataList)
	distanceList.append(targetDistanceData)
    # Area Plot
	area = np.empty([num_iteration,2])
	for i in range(num_iteration):
		shapeData = np.empty([num_wolves,2])
		for j in range(len(dataList)):
			shapeData[j,:] = dataList[j][i,1:]
		hull = ConvexHull(shapeData)
		area[i,:] = np.array([i,math.log(hull.area)])		
	convexHullList.append(area)

	#Swarm area plot
	centroidData = np.empty([num_iteration,3])
	for i in range(num_iteration):
		centroidData[i,0] = i
		buff = np.array([0,0])
		for wolf in dataList:
			buff = buff+ wolf[i,1:]
		buff = buff/num_wolves
		centroidData[i,1] = buff[0]
		centroidData[i,2] = buff[1]	
	centroidList.append(centroidData)
	# num_wolves = num_wolves + 1

#Plot the centroid data
fig = plt.figure()
values = list(range(1,14))
colors = itertools.cycle(['red', 'orange', 'green', 'blue', 'violet','cyan','#2300A8','#00A658'])
for i in range(1,14):
	ax = fig.add_subplot(4, 4, i, projection='3d')
	ax.scatter3D(centroidList[i-1][:,0],centroidList[i-1][:,1],centroidList[i-1][:,2], color = next(colors), edgecolors = 'black')
	ax.set_title('SLL = ' + str(values[i-1]))
	ax.set_xlabel('Iteration Number')
	# ax.set_ylabel('X coor')
	# ax.set_zlabel('Y Coordinate')
# plt.title("Scatter plot of Swarm centroid for different values of the Exploration factor")
fig.suptitle("Scatter plot of Swarm centroid for different social learning (SLL) numbers")
plt.show()

#Plot the swarm centroid plot
fig = plt.figure()
values = list(range(1,14))
# colors = itertools.cycle(['red', 'orange', 'green', 'blue', 'violet','cyan','#2300A8','#00A658'])
index = 0
for area in convexHullList:
	plt.plot(area[:,0],area[:,1], '-', label = 'SLL = ' + str(values[index]))
	index = index+1
plt.legend(loc='best')
plt.xlabel('Iteration Number')
plt.ylabel('Log Area of convex hull of swarm')
plt.title('Log Area Plot of the Convex Hull covering the whole swarm for different social learning numbers')
plt.show()

#Plot the average distance plot
fig = plt.figure()
values = list(range(1,14))
# colors = itertools.cycle(['red', 'orange', 'green', 'blue', 'violet','cyan','#2300A8','#00A658'])
index = 0
for targetDistanceData in distanceList:
	plt.plot(targetDistanceData[:,0],targetDistanceData[:,1], '-', label = 'SLL = ' + str(values[index]))				
	index = index+1
plt.legend(loc='best')
plt.xlabel('Iteration Number')
plt.ylabel('Average distance of a robot from the target')
plt.title('Average distance of the robot from target for different social learning numbers')
plt.show()
