import numpy as np
import matplotlib.pyplot as plt
import math
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import colors
from io import StringIO

path1 = '/Users/anikethramesh/Desktop/Stuff/Programming/ProjectFolder/Swarm_Trial1/controllers/wolf_trial2/'
folderNames = [('Webots_'+str(i)+'/') for i in range(1,11)]
folderPaths = [(path1+folder) for folder in folderNames]

index = 1
fig = plt.figure()

for directory in folderPaths:
	print(directory)
	files = [filename for filename in os.listdir(directory)]
	robot_data = []
	for file in files:
		print(file)
		data = np.genfromtxt(raw_string(file), delimiter = ',')
	fig.add_subplot(3,3,index)
	for data in robot_data:
		print(len(data))
		plt.plot(data[:,1],data[:,2],'o')
		plt.xlabel('X Coordinates')
		plt.ylabel('Y  Coordinates')
		plt.title('Parameter Set No.' + str(index))
plt.show()