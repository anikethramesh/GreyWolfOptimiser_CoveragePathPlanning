import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d

num_wolves = 15
# num_iteration = 500
path = './Python_TargetChange_Reset/Trial_1/wolf'
suff = '.csv'
files = [(path+str(i)+suff) for i in range(num_wolves)]


data_list = []

for f in files:
	data = np.genfromtxt(f, delimiter=',')
	data_list.append(data)

distanceList = []

for data in data_list:
	targetDistanceData = np.zeros([len(data),2])
	for i in range(len(data)):
		targetDistanceData[i,0] = i
		target = np.array([0,0])
		if(i<127):
			target = np.array([593,593])
		elif(i>=127 and i<389):
			target = np.array([-493,-493])
		elif(i>=389):
			target = np.array([-1000,1000])
		targetDistanceData[i,1] = math.log(np.linalg.norm(target - data[i,1:]))
	distanceList.append(targetDistanceData)


fig = plt.figure()
index = 1
for distance in distanceList:
	fig.add_subplot(3, 5, index)
	plt.plot(distance[:,0],distance[:,1])
	plt.ylabel('Log of Distance from target')
	plt.xlabel('Iteration number')
	plt.title('Wolf' + str(index-1))	
	index = index+1

plt.suptitle('Target Changing in Python - (593,593),(493,493), (393,393) in iterations 127 and 389')
plt.show()




# for folder in folders:
# 	dataList = []
# 	for i in range(num_wolves):
# 		data = np.genfromtxt(folder + path2 +str(i)+path3, delimiter = ',')
# 		dataList.append(data)

# 	index = 0
# 	for data in dataList:		
# 		ax = plt.axes(projection='3d')
# 		ax.view_init(azim=-65,elev = 15)
# 		ax.scatter3D(data[:,0],data[:,1],data[:,2])
# 		ax.set_title('Position of Robot '+str(index)+' over '+ str(num_iteration)+' Iterations')
# 		ax.set_xlabel('Iteration Number')
# 		ax.set_ylabel('X Coordinate')	
# 		ax.set_zlabel('Y Coordinate')
# 		# plt.savefig(path + 'wolf' +str(index)+'_trajectoryScatter.png',bbox_inches='tight', dpi=300)
# 		index = index+1
# 		plt.show()

# #Average distance Plot
# 	index = 0
# 	for wolf in dataList:
# 		targetDistanceData = np.empty([num_iteration,2])	
# 		for i in range(num_iteration):
# 			targetDistanceData[i,0] = wolf[i,0]
# 			targetDistanceData[i,1] = math.log(np.linalg.norm(np.array([593,593]) - wolf[i,1:]))
# 		plt.plot(targetDistanceData[:,0],targetDistanceData[:,1])
# 		plt.xlabel('Iteration Number')
# 		plt.ylabel('Distance from target')
# 		plt.title('Distance of Robot ' + str(index)+ ' from target'+' over '+ str(num_iteration)+' Iterations')
# 		plt.savefig(path + 'wolf' +str(index)+'_distanceFromTarget.png', dpi=300)
# 		index = index+1
# 		plt.clf()
