# import csv

# with open('wolf0.txt') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter = ',')
# 	for row in csv_reader:
# 		print(row)


import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d

num_wolves = 15
num_iteration = 300
path1 = './Data_PythonSimulations/socialLearning/python_300_'
path2 = '_4_0.5_15/'
pathList = [(path1 + str(val) + path2) for val in range(1,14)]

for path in pathList:
	dataList = []
	#Import all the files as separate arrays
	for i in range(num_wolves):
		data = np.genfromtxt(path + 'wolf' + str(i)+'.csv', delimiter = ',')
		dataList.append(data)

	#Plot graphs of the trajectories for each of them.
	# from mpl_toolkits.mplot3d import Axes3D
	index = 0
	for data in dataList:		
		ax = plt.axes(projection='3d')
		ax.view_init(azim=-65,elev = 15)
		ax.scatter3D(data[:,0],data[:,1],data[:,2])
		ax.set_title('Position of Robot '+str(index)+' over '+ str(num_iteration)+' Iterations')
		ax.set_xlabel('Iteration Number')
		ax.set_ylabel('X Coordinate')	
		ax.set_zlabel('Y Coordinate')
		plt.savefig(path + 'wolf' +str(index)+'_trajectoryScatter.png',bbox_inches='tight', dpi=300)
		index = index+1
		plt.clf()	

	#Plot graphs of distance from target
	index = 0
	for wolf in dataList:
		targetDistanceData = np.empty([num_iteration,2])	
		for i in range(num_iteration):
			targetDistanceData[i,0] = wolf[i,0]
			targetDistanceData[i,1] = math.log(np.linalg.norm(np.array([593,593]) - wolf[i,1:]))
		plt.plot(targetDistanceData[:,0],targetDistanceData[:,1])
		plt.xlabel('Iteration Number')
		plt.ylabel('Distance from target')
		plt.title('Distance of Robot ' + str(index)+ ' from target'+' over '+ str(num_iteration)+' Iterations')
		plt.savefig(path + 'wolf' +str(index)+'_distanceFromTarget.png', dpi=300)
		index = index+1
		plt.clf()

	#Average distance Plot
	index = 0
	targetDistanceData = np.zeros([num_iteration,2])		
	for wolf in dataList:
		for i in range(num_iteration):
					targetDistanceData[i,0] = wolf[i,0]
					targetDistanceData[i,1] += math.log(np.linalg.norm(np.array([593,593]) - wolf[i,1:]))/len(dataList)

	#Centroid Plot
	centroidData = np.empty([num_iteration,3])
	for i in range(num_iteration):
		centroidData[i,0] = i
		buff = np.array([0,0])
		for wolf in dataList:
			buff = buff+ wolf[i,1:]
		buff = buff/num_wolves
		centroidData[i,1] = buff[0]
		centroidData[i,2] = buff[1]	
	ax = plt.axes(projection='3d')
	ax.view_init(azim=-65,elev = 15)
	ax.scatter3D(centroidData[:,0],centroidData[:,1],centroidData[:,2])
	ax.set_title('Centroid of Robot Positions' + ' over '+ str(num_iteration)+' Iterations')
	ax.set_xlabel('Iteration Number')
	ax.set_ylabel('X Coordinate')	
	ax.set_zlabel('Y Coordinate')
	plt.savefig(path + 'Centroid Plot.png',bbox_inches='tight', dpi=300)
	plt.clf()

	#Intraswarm area plot
	# from scipy.spatial import ConvexHull, convex_hull_plot_2d
	fig = plt.figure()
	# ax = fig.add_subplot(111,projection='3d')
	area = np.empty([num_iteration,2])
	for i in range(num_iteration):
		shapeData = np.empty([num_wolves,2])
		for j in range(len(dataList)):
			shapeData[j,:] = dataList[j][i,1:]
		hull = ConvexHull(shapeData)
		# for simplex in hull.simplices
		# 	ax.plot(shapeData[simplex, 0], shapeData[simplex, 1],i,'k-')
		area[i,:] = np.array([i,math.log(hull.area)])
	plt.plot(area[:,0],area[:,1], '-o')

	plt.xlabel('Iteration Number')
	plt.ylabel('Area of convex hull of swarm')

	plt.title('Log Area Plot of the Convex Hull covering the whole swarm')
	plt.savefig(path + 'wolf' +str(index)+'_convexHullAreaPlot.png', dpi=300)
	# plt.show()
	plt.clf()

	# Convex Hull PLot
	# from scipy.spatial import ConvexHull, convex_hull_plot_2d
	# fig = plt.figure()
	# ax = fig.add_subplot(111,projection='3d')
	# area = np.empty([num_iteration,2])
	# for i in range(num_iteration):
	# 	shapeData = np.empty([15,2])
	# 	for j in range(len(dataList)):
	# 		shapeData[j,:] = dataList[j][i,1:]
	# 	hull = ConvexHull(shapeData)
	# 	for simplex in hull.simplices:
	# 		ax.plot(shapeData[simplex, 0], shapeData[simplex, 1],i,'k-')
	# ax.set_title('Convex Hull covering the whole swarm for each iteration')
	# ax.set_xlabel('X_coordinates of convex hull')
	# ax.set_ylabel('Y_coordinates of convex hull')

	# x_scale=1
	# y_scale=1
	# z_scale=4

	# scale=np.diag([x_scale, y_scale, z_scale, 1.0])
	# scale=scale*(1.0/scale.max())
	# scale[3,3]=1.0

	# def short_proj():
	#   return np.dot(Axes3D.get_proj(ax), scale)

	# ax.get_proj=short_proj


	# ax.tick_params(axis='z', which='major', pad=15)

	ax.set_zlabel('Iteration Number')
	ax.view_init(azim=-65,elev = 15)
	plt.savefig(path + 'Convex Hull Plot', dpi=300)
	# num_wolves = num_wolves + 1
	# plt.show()

