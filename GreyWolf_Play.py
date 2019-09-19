import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import math
import csv
import os


class Visualization:
	def __init__(self,pauseTiming, x_lim_lower, x_lim_upper, y_lim_lower, y_lim_upper):
		self.x_lim_lower = x_lim_lower
		self.x_lim_upper = x_lim_upper
		self.y_lim_lower = y_lim_lower
		self.y_lim_upper = y_lim_upper
		self.pause_timing = pauseTiming
		self.target = np.array([593,593])

	def plot_list(self,wolfList):
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		plt.plot(self.target[0],self.target[1],'x')
		plt.axis([self.x_lim_lower,self.x_lim_upper,self.y_lim_lower,self.y_lim_upper])		
		for wolf in wolfList:
			plt.plot(wolf.currentPosition[0],wolf.currentPosition[1],'o')
		plt.pause(self.pause_timing)
		# plt.show()
		plt.clf()

#Class for individual wolf
class GreyWolf:
	def __init__(self,startPosition,num_iter,index,fileName, exCon, speed):
		self.currentPosition = startPosition
		self.nextPosition = self.currentPosition
		self.isAlpha = False
		self.isBeta = False
		self.isDelta = False
		self.fitness = 0
		self.speed_factor = speed
		self.explorationConstant = exCon
		self.communicationRange = 10
		self.num_iter = num_iter
		# self.index = index
		filename = fileName+"wolf"+str(index)+".csv"
		os.makedirs(os.path.dirname(filename), exist_ok = True)
		self.logFile = open(filename, "w+")
		self.logWriter = csv.writer(self.logFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	# def return_currentPosition(self):
	# 	return self.currentPosition

	def calculateConstants(self,current_iter):
		a = self.explorationConstant*(1 - current_iter/self.num_iter)*np.array([1,1])
		r1 = np.random.rand(2)
		r2 = np.random.rand(2)
		# print("a is", a,"\n")
		return a, r1, r2

	def reset_Leadership(self):
		self.isAlpha = False
		self.isBeta = False
		self.isDelta = False

	def makeAlpha(self):
		self.isAlpha = True

	def makeBeta(self):
		self.isBeta = True

	def makeDelta(self):
		self.isDelta = True

	def isOmega(self):
		if(self.isAlpha == False and self.isBeta == False and self.isDelta == False):
			return True
		else:
			return False

	def updateCurrentPosition(self,current_iter):
		self.currentPosition = self.currentPosition + self.speed_factor*((self.nextPosition - self.currentPosition))
		self.logWriter.writerow([current_iter, self.currentPosition[0], self.currentPosition[1]])
		# self.logFile.write(str(self.currentPosition)+"\n")
		# self.logFile.write("\n")

	def calculateNewPosition(self,position_target,leadershipList, current_iter):
		self.nextPosition = np.array([0.,0.])
		# a,r1,r2 = self.calculateConstants(current_iter)
		# A = 2*np.multiply(a,r1) - a
		# C = 2*r2
		if(self.isOmega):
			a,r1,r2 = self.calculateConstants(current_iter)
			A = 2*np.multiply(a,r1) - a
			C = 2*r2			
			for leader in leadershipList:
				D = np.multiply(C,leader.currentPosition) - self.currentPosition
				self.nextPosition += (leader.currentPosition - np.multiply(A,D))/len(leadershipList)
			# self.nextPosition = self.nextPosition/len(leadershipList)
		else:
			a,r1,r2 = self.calculateConstants(current_iter)
			A = 2*np.multiply(a,r1) - a
			C = 2*r2			
			D = np.multiply(C,position_target) - self.currentPosition
			self.nextPosition = (position_target - np.multiply(A,D))
		# if(self.isAlpha):	
		self.updateCurrentPosition(current_iter)

#Class for swarm level characteristics
class swarmControl:
	def __init__(self, num_iter, num_wolves, social_num):
		self.num_iterations = num_iter
		self.social_learning_number = social_num
		self.position_target = np.array([593,593])
		self.position_start = np.array([0,0])
		self.num_wolves = num_wolves
		# self.wolf_positions = np.empty([self.num_wolves,2])
	
	def set_target(self,position_target):
		self.position_target = position_target

	def generate_startingPositions(self):
		x_coor = self.position_start[0]+10*np.cos(np.linspace(0,2*math.pi,self.num_wolves))
		y_coor = self.position_start[1]+10*np.sin(np.linspace(0,2*math.pi,self.num_wolves))
		return np.transpose(np.array([x_coor,y_coor]))

	# def get_wolfPositions(self, wolfList):
	# 	for i in range(self.num_wolves):
	# 		self.wolf_positions[i,:] = wolfList[i].currentPosition

	def set_leadership(self, wolfList):
		wolfList = sorted(wolfList, key = lambda x: x.fitness)
		wolfList[-1].makeAlpha()
		wolfList[-2].makeBeta()
		wolfList[-3].makeDelta()

		return [wolfList[-1],wolfList[-2],wolfList[-3]]		

	def calculateNextIteration(self, wolfList, current_iter):
		leadershipList = self.set_leadership(wolfList)
		for wolf in wolfList:
			wolf.calculateNewPosition(self.position_target,leadershipList,current_iter)
			wolf.reset_Leadership()

#Class dealing with the visualization of the robots

# Get the axes limits and setup axis object.
# For plotting get the robot positions in an np array
# Each time the positions change, there must be a function to update the plot

#Class dealing with handling the fitness related properties and methods.
class fitness:
	def __init__(self,k_intraSwarm, k_coverage, k_goalDistance, num_iter):
		self.k_intraSwarmDistance  = k_intraSwarm
		self.k_individualCoverage = k_coverage
		self.k_distanceFromGoal = k_goalDistance
		self.num_iter = num_iter

	def calculateFitness(self, wolfList, current_iter, position_target, position_start):				
		self.k_intraSwarmDistance = 1 - (current_iter/self.num_iter)
		self.k_coverageIndividual = 1 - (current_iter/self.num_iter)
		self.k_distanceFromGoal = 1 + (current_iter/self.num_iter)
		
		for i in range(len(wolfList)):
			coverageIndividual = np.linalg.norm(position_start - wolfList[i].currentPosition)
			distanceFromGoal = np.linalg.norm(position_target - wolfList[i].currentPosition)			
			intraSwarmDistance = 0
			for j in range(len(wolfList)):
				if(i!=j):
					intraSwarmDistance = intraSwarmDistance+ (wolfList[i].communicationRange/np.linalg.norm(wolfList[i].currentPosition - wolfList[j].currentPosition))
			wolfList[i].fitness = (self.k_coverageIndividual*coverageIndividual) + (self.k_intraSwarmDistance*intraSwarmDistance) - (self.k_distanceFromGoal*distanceFromGoal)
			print(wolfList[i].fitness)

def calculateFileName(experiment_type,iterations,social_num, exploration, multiplication, num_robots):
	return "./"+experiment_type+"/python_"+str(iterations)+"_"+str(social_num)+"_"+str(exploration)+"_"+str(multiplication)+"_"+str(num_robots)+"/"

##############################
if __name__ == '__main__':

	np.random.seed()
	plotter = Visualization(0.01,-2000,2000,-2000,2000)
	#Default Settings
	
	number_iterations = 300
	social_number = 3
	explorationConstant = 4
	multiplicationConstant = 0.5

	#Number of robots experiments
	
	for number_of_robots in range(4,20):
		swarm = swarmControl(number_iterations,number_of_robots,social_number)
		fitnessValue = fitness(1,1,1,number_iterations)
		#initialize grey wolves
		wolves = []
		index = 0
		fileName = calculateFileName("numberOfRobots",number_iterations, social_number, explorationConstant,multiplicationConstant, number_of_robots)
		for i in swarm.generate_startingPositions():
			wolves.append(GreyWolf(i,number_iterations,index,fileName,explorationConstant,multiplicationConstant))
			index= index+1
		#Loop
		for i in range(number_iterations):
			fitnessValue.calculateFitness(wolves, i, swarm.position_target, swarm.position_start)
			# print("Fitness called")
			swarm.calculateNextIteration(wolves,i)
			# print("Iteration called")
			# print(wolves[1].currentPosition)
			# plotter.plot_list(wolves)
			# print("Iteration Number", i)

	#Social Learning Number Experiment
	number_of_robots = 15
	number_iterations = 300
	# social_number = 3
	explorationConstant = 4
	multiplicationConstant = 0.5	
	for social_number in range(1,14):
		swarm = swarmControl(number_iterations,number_of_robots,social_number)
		fitnessValue = fitness(1,1,1,number_iterations)
		#initialize grey wolves
		wolves = []
		index = 0
		fileName = calculateFileName("socialLearning",number_iterations, social_number, explorationConstant,multiplicationConstant, number_of_robots)
		for i in swarm.generate_startingPositions():
			wolves.append(GreyWolf(i,number_iterations,index,fileName,explorationConstant,multiplicationConstant))
			index= index+1
		#Loop
		for i in range(number_iterations):
			fitnessValue.calculateFitness(wolves, i, swarm.position_target, swarm.position_start)
			# print("Fitness called")
			swarm.calculateNextIteration(wolves,i)
			# print("Iteration called")
			# print(wolves[1].currentPosition)
			# plotter.plot_list(wolves)
			# print("Iteration Number", i)

	# Multiplication Factor Experiment
	number_of_robots = 15
	number_iterations = 300
	social_number = 3
	explorationConstant = 4
	# multiplicationConstant = 0.5
	for multiplicationConstant in np.linspace(0.01,2,20):
		swarm = swarmControl(number_iterations,number_of_robots,social_number)
		fitnessValue = fitness(1,1,1,number_iterations)
		#initialize grey wolves
		wolves = []
		index = 0
		fileName = calculateFileName("MultiplicationFactor",number_iterations, social_number, explorationConstant,multiplicationConstant, number_of_robots)
		for i in swarm.generate_startingPositions():
			wolves.append(GreyWolf(i,number_iterations,index,fileName,explorationConstant,multiplicationConstant))
			index= index+1
		#Loop
		for i in range(number_iterations):
			fitnessValue.calculateFitness(wolves, i, swarm.position_target, swarm.position_start)
			# print("Fitness called")
			swarm.calculateNextIteration(wolves,i)
			# print("Iteration called")
			# print(wolves[1].currentPosition)
			# plotter.plot_list(wolves)
			# print("Iteration Number", i)
	
	#Exploration Factor Experiments
	number_of_robots = 15
	number_iterations = 300
	social_number = 3
	# explorationConstant = 4
	multiplicationConstant = 0.5	
	for explorationConstant in range(2,12):
		swarm = swarmControl(number_iterations,number_of_robots,social_number)
		fitnessValue = fitness(1,1,1,number_iterations)
		#initialize grey wolves
		wolves = []
		index = 0
		fileName = calculateFileName("ExplorationFactor",number_iterations, social_number, explorationConstant,multiplicationConstant, number_of_robots)
		for i in swarm.generate_startingPositions():
			wolves.append(GreyWolf(i,number_iterations,index,fileName,explorationConstant,multiplicationConstant))
			index= index+1
		#Loop
		for i in range(number_iterations):
			fitnessValue.calculateFitness(wolves, i, swarm.position_target, swarm.position_start)
			# print("Fitness called")
			swarm.calculateNextIteration(wolves,i)
			# print("Iteration called")
			# print(wolves[1].currentPosition)
			# plotter.plot_list(wolves)
			# print("Iteration Number", i)
	
	# Iterations Experiments
	number_of_robots = 15
	# number_iterations = 300
	social_number = 3
	explorationConstant = 4
	multiplicationConstant = 0.5	
	for number_iterations in range(50,550,50):
		swarm = swarmControl(number_iterations,number_of_robots,social_number)
		fitnessValue = fitness(1,1,1,number_iterations)
		#initialize grey wolves
		wolves = []
		index = 0
		fileName = calculateFileName("NumberIterations",number_iterations, social_number, explorationConstant,multiplicationConstant, number_of_robots)
		for i in swarm.generate_startingPositions():
			wolves.append(GreyWolf(i,number_iterations,index,fileName,explorationConstant,multiplicationConstant))
			index= index+1
		#Loop
		for i in range(number_iterations):
			fitnessValue.calculateFitness(wolves, i, swarm.position_target, swarm.position_start)
			# print("Fitness called")
			swarm.calculateNextIteration(wolves,i)
			# print("Iteration called")
			# print(wolves[1].currentPosition)
			# plotter.plot_list(wolves)
			# print("Iteration Number", i)	