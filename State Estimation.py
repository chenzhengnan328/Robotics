#!/usr/bin/env python3

# Columbia Engineering
# MECS 4602 - Fall 2018

import math
import numpy
import time

import rospy

from state_estimator.msg import RobotPose
from state_estimator.msg import SensorData

class Estimator(object):
	def __init__(self):

        # Publisher to publish state estimate
		self.pub_est = rospy.Publisher("/robot_pose_estimate", RobotPose, queue_size=1)

        # Initial estimates for the state and the covariance matrix
		self.x = numpy.zeros((3,1))
		self.P = numpy.zeros((3,3))

        # Covariance matrix for process (model) noise
		self.V = numpy.zeros((3,3))
		self.V[0,0] = 0.0025
		self.V[1,1] = 0.0025
		self.V[2,2] = 0.005
		self.step_size = 0.01
		self.r = 0.1
		self.b = 0.05

        # Subscribe to command input and sensory output of robot
		rospy.Subscriber("/sensor_data", SensorData, self.sensor_callback)

    # This function gets called every time the robot publishes its control
    # input and sensory output. You must make use of what you know about
    # extended Kalman filters to come up with an estimate of the current
    # state of the robot and covariance matrix.
    # The SensorData message contains fields 'vel_trans' and 'vel_ang' for
    # the commanded translational and rotational velocity respectively.
    # Furthermore, it contains a list 'readings' of the landmarks the
    # robot can currently observe
	def Jacobian(self, xr,xl, yr, yl):
		H[2*i][0]=(x_pred-xl[i])/math.sqrt( math.pow((x_pred - xl[i]), 2) + math.pow((y_pred - yl[i]), 2) )
		H[2*i][1]=(y_pred-yl[i])/math.sqrt( math.pow((x_pred - xl[i]), 2) + math.pow((y_pred - yl[i]), 2) )
		H[2*i+1][0]=(yl[i] - y_pred)/(math.pow((xl[i]-x_pred), 2) + math.pow((yl[i]-y_pred), 2) )
		H[2*i+1][1]=(x_pred - xl[i])/(math.pow((xl[i]-x_pred), 2) + math.pow((yl[i]-y_pred), 2) )    		
		
		return H
		
	def estimate(self, sens):

	#### ----- YOUR CODE GOES HERE ----- ####
		t = self.step_size
		x_pred = self.x[0] + t * sens.vel_trans * math.cos(self.x[2])
		y_pred = self.x[1] + t * sens.vel_trans * math.sin(self.x[2])
		theta_pred = self.x[2] +t * sens.vel_ang
		
		state =numpy.array([x_pred, y_pred, theta_pred])
		
		xdx = -t*sens.vel_trans*math.sin(self.x[2])
		ydy = t*sens.vel_trans*math.cos(self.x[2])
		
		F = [[1,0,((-t)*sens.vel_trans*math.sin(self.x[2]))], [0, 1, (t*sens.vel_trans*math.cos(self.x[2]))], [0,0,1]]
		
		P_pred = numpy.dot(numpy.dot(F, self.P), numpy.transpose(F)) +self.V

		
		
		xl, yl, y, rl = [], [], [], []

		for i in range(len(sens.readings)):
			object_p = numpy.array([sens.readings[i].landmark.x,sens.readings[i].landmark.y])
			robot_p = numpy.array([x_pred[0],y_pred[0]])
			
			
			distance = numpy.linalg.norm(robot_p - object_p)
			
			if distance > 0.1:
				xl.append(sens.readings[i].landmark.x)
				yl.append(sens.readings[i].landmark.y)
				y.append(sens.readings[i].range)
				y.append(sens.readings[i].bearing)
			
		m = 2 *len(xl)
		n = len(self.x)				
		H = numpy.zeros((m,n))
		W = numpy.zeros((m,m))
		y_p = numpy.zeros((m,1))
		
		#(x_pred-xl)/math.sqrt( math.pow((x_pred - xl), 2) + math.pow((y_pred - yl), 2) )
		#(y_pred-yl)/math.sqrt( math.pow((x_pred - xl), 2) + math.pow((y_pred - yl), 2) )
		#(yl - y_pred)/(math.pow((xl-x_pred), 2) + math.pow((yl-y_pred), 2) )
		#(x_pred - xl)/(math.pow((xl-x_pred), 2) + math.pow((yl-y_pred), 2) )
		
		if len(xl):
			for i in range(len(xl)):	
					
				H[2*i][0]=(x_pred-xl[i])/math.sqrt( math.pow((x_pred - xl[i]), 2) + math.pow((y_pred - yl[i]), 2) )
				H[2*i][1]=(y_pred-yl[i])/math.sqrt( math.pow((x_pred - xl[i]), 2) + math.pow((y_pred - yl[i]), 2) )
				H[2*i+1][0]=(yl[i] - y_pred)/(math.pow((xl[i]-x_pred), 2) + math.pow((yl[i]-y_pred), 2) )
				H[2*i+1][1]=(x_pred - xl[i])/(math.pow((xl[i]-x_pred), 2) + math.pow((yl[i]-y_pred), 2) )
				H[2*i+1][2] =-1
				#print(H)
				W[2*i][2*i] = self.r
				W[2*i+1][2*i+1] = self.b
				
				y_p[2*i] = math.sqrt((x_pred-xl[i])*(x_pred - xl[i])+(y_pred - yl[i])*(y_pred - yl[i]))
				y_p[2*i+1] = math.atan2(yl[i]-y_pred, xl[i]-x_pred) -theta_pred
				
			
			nu = numpy.array(y) - numpy.transpose(y_p)
			
			for i in range(len(xl)):
				while nu[0][2*i+1] < -numpy.pi:# math.pi:
					nu[0][2*i+1] += 2 * numpy.pi
					
				while nu[0][2*i+1] > numpy.pi:
				
					nu[0][2*i+1] -= 2*numpy.pi

		
			nu = numpy.transpose(nu)
			S = numpy.dot(numpy.dot(H,P_pred),numpy.transpose(H))+ W
			R = numpy.dot(numpy.dot(P_pred, numpy.transpose(H)), numpy.linalg.pinv(S))
			self.x = state + numpy.dot(R,nu)
			self.P = P_pred -numpy.dot(numpy.dot(R,H), P_pred)
	
		else:
			self.x = state
			self.P = P_pred

        #### ----- YOUR CODE GOES HERE ----- ####

	def sensor_callback(self,sens):

        # Publish state estimate
		self.estimate(sens)
		est_msg = RobotPose()
		est_msg.header.stamp = sens.header.stamp
		est_msg.pose.x = self.x[0]
		est_msg.pose.y = self.x[1]
		est_msg.pose.theta = self.x[2]
		self.pub_est.publish(est_msg)

if __name__ == '__main__':
	rospy.init_node('state_estimator', anonymous=True)
	est = Estimator()
	rospy.spin()
