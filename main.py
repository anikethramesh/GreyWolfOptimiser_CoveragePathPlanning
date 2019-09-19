import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn
import numpy as np
import time
import math

position_target = np.array([592,592])
position_Start = np.array([0,0])
numberOfWolves = 15
socialLearningNumber = 3
num_iter = 300
#This is a function to change the position of the target/prey and reset the iteration count.
def changePreyPosition(x_pos, y_pos):
	global position_target
	global i
	position_target = np.array([x_pos,y_pos])
	i=0

#Handling the mouse. This is used to track the mouse and update the position of prey accordingly.
class ClickChecker:
	def __init__(self,preyPoint):
		self.preyPoint = preyPoint
		self.cid = preyPoint.figure.canvas.mpl_connect('button_press_event', self)
	def __call__(self, event):
		print(event.xdata, event.ydata)
		changePreyPosition(event.xdata, event.ydata)

#Social learning number determines the number of wolves that the omegas learn from. In this case, it is 3 - alpha wolf, beta wolf and a gamma wolf

#This is used to calculate the constants that will be used throughout the program. 
def calculateConstants(current_iter):

	#a is a constant linearly decrease from 4 to 0. When a>1 exploration is preffered, and otherwise exploitation is preferred.
	a = 4*(1 - current_iter/num_iter)*np.array([1,1])
	# a = np.array([3,3])
	r1 = np.random.rand(2)
	# r1 = np.array([0.025,0.462])
	r2 = np.random.rand(2)
	# r2 = np.array([0.067,0.9])
	return a, r1, r2	

	# print(position_target)

def calculateNewPosition(prey_position,currentPosition, current_iter, fitnessArray, isOmega):
	#Omegas learn from the alpha beta and the delta. Their next position is determined by the next position of the these three wolves
	#IsOmega - marker for each wolf that determines whether they are learning from the leaders or going behind the prey.
	# scalingVector = np.array([1/20,1/20])
	if(isOmega==True):
		nextPosition = np.array([0.,0.])
		for i in range(socialLearningNumber):
			position = wolfPositions[fitnessArray[-(socialLearningNumber+1)][0],:]
			a,r1,r2 = calculateConstants(current_iter)
			A = 2*np.multiply(a,r1) - a
			C = 2*r2
			D = np.multiply(C,position) - currentPosition
			# D = np.multiply(scalingVector,D)
			nextPosition += (position - np.multiply(A,D))/socialLearningNumber
			# print("Magnitude of nextPos", np.linalg.norm(nextPosition),"\n")		
	else:
		a,r1,r2 = calculateConstants(current_iter)
		A = 2*np.multiply(a,r1) - a
		C = 2*r2
		D = np.multiply(C,prey_position) - currentPosition
		# D = np.multiply(scalingVector,D)
		nextPosition = (prey_position - np.multiply(A,D))

	#Vary the multiplication factor from 0.01 to 1 to observe different swarm behaviours.
	# 0.01 - 0.1 gives movement similar to a swarm lead by an alpha, beta and delta
	# whereas a multiplication factor in the range of 0.5 to 1 makes it work like an optimizer.
	multiplicationFactor = 0.3
	# buffer = currentPosition + multiplicationFactor*((nextPosition - currentPosition)/np.linalg.norm(nextPosition - currentPosition))
	buffer = currentPosition + multiplicationFactor*(nextPosition - currentPosition)
	return buffer

def fitnessFunction(currentPosition, wolfPositions,current_iter):

	intraSwarmDistance = 0
	# There is a fitness associated with the distance of each robot from other members of the swarm. 
	# This is decreased over the number of iterations to give more preference to encircling and attacking behaviours.
	k_intraSwarmDistance =1 - (current_iter/num_iter)
	#This would be varied in the simulation environment.Communication threshold penalizes the wolves if they move further 
	#away from each other than their communication ranges allow.
	wolf_communicationThreshold = 10
	for wolf in range(len(wolfPositions)):
			print("currentPosition",np.linalg.norm(wolf - currentPosition))
			intraSwarmDistance += wolf_communicationThreshold/np.linalg.norm(wolf - currentPosition)
	#Higher individual coverage  increases reward in the beginning, and is steadily decreased over iterations.
	k_coverageIndividual = 1 - (current_iter/num_iter)
	#This reward is increased over iterations. 
	k_distanceFromGoal = 1 + (current_iter/num_iter)
	coverageIndividual = np.linalg.norm(position_Start - currentPosition)
	distanceFromGoal = np.linalg.norm(position_target - currentPosition)
	#total fitness
	fitness = (k_coverageIndividual*coverageIndividual) + (k_intraSwarmDistance*intraSwarmDistance) - (k_distanceFromGoal*distanceFromGoal)
	print(fitness)
	return fitness
 
#Calculates the fitness for all the wolves.
def calculateBest(wolfPositions,current_iter):
	fitnessArray = [] 
	for i in range(len(wolfPositions)):
		fitnessArray.append([i, fitnessFunction(wolfPositions[i,:],wolfPositions,current_iter)])
	return fitnessArray

#initialization for starting  position
x_coor = position_Start[0]+np.cos(np.linspace(0,2*math.pi,numberOfWolves))
y_coor = position_Start[1]+np.sin(np.linspace(0,2*math.pi,numberOfWolves))

wolfPositions = np.array([x_coor,y_coor])
wolfPositions = np.transpose(wolfPositions)

np.random.seed()
fig = plt.figure()
ax = fig.add_subplot(111)
prey, = ax.plot(position_target[0],position_target[1],'x')
mouse = ClickChecker(prey)
i = 0
# for i in range(num_iter):
while i in range(num_iter):
	fitnessArray = calculateBest(wolfPositions,i)
	fitnessArray = sorted(fitnessArray, key = lambda x: x[1])
	#Create an array that sorts the wolves by their fitness. 
	#The last three elements are respectively the delta, beta and the alpha.
	print("\nHighest Fitness Position", wolfPositions[fitnessArray[-1][0],:], "\t Score of fittest", fitnessArray[-1][1])

	for f in range(len(fitnessArray)):
		if(f<numberOfWolves - socialLearningNumber):
			wolfPositions[fitnessArray[f][0],:] = calculateNewPosition(position_target,wolfPositions[f,:],i,fitnessArray,True)
		else:
			wolfPositions[fitnessArray[f][0],:] = calculateNewPosition(position_target,wolfPositions[f,:],i,fitnessArray,False)

	plt.plot(position_target[0],position_target[1],'x')
	for j in wolfPositions:
		#If the robobts leave the plot, you may have to change the axes limits to get all of them inside the frame.
		plt.axis([-2000,2000,-2000,2000])		
		plt.plot(j[0],j[1],'o')
	plt.pause(0.01)
	plt.clf()
	print(i,"\n")
	i+=1
plt.show()