from arm_dynamics_base import ArmDynamicsBase
import numpy as np
from geometry import rot, xaxis, yaxis
import math
class ArmDynamicsStudent(ArmDynamicsBase):

	def dynamics_step(self, state, action, dt):
        # state has the following format: [q_0, ..., q_(n-1), qdot_0, ..., qdot_(n-1)] where n is the number of links
        # action has the following format: [mu_0, ..., mu_(n-1)]
        # You can make use of the additional variables:
        # self.num_links: the number of links
        # self.joint_viscous_friction: the coefficient of viscous friction
        # self.link_lengths: an array containing the lengths of all the links
        # self.link_masses: an array containing the masses of all the links
        # known q0, mu, q0dot
        # unknown qn, qdot, qdotdot, f, a
		result = np.zeros(6*self.num_links)
		constrain = np.zeros((6 * self.num_links, 6 * self.num_links))
		w = 0 
		
		if self.num_links == 3:
			w0 = state[self.num_links]
			q0dot = state[self.num_links]
			#print(math.sin(state[0]), math.cos(state[0]))
			constrain[:2, :2] = [[1,0],[0,1]]
			constrain[:2,2:4] = [ [-math.cos(-state[1]), math.sin(-state[1]) ], [ -math.sin(-state[1]), -math.cos(-state[1]) ] ]
			constrain[:2,6:8] = [ [ -self.link_masses[0], 0 ],[ 0, -self.link_masses[0] ] ]
			constrain[1][12] = -self.link_masses[0] * 0.5 * self.link_lengths[0]
			f1 = np.array( [ -self.link_masses[0]*0.5*self.link_lengths[0]*pow(w0, 2),  0 ] )
			f2 = np.array( [ self.link_masses[0] * 9.8 * math.sin(state[0]), -self.link_masses[0] * 9.8 * math.cos(state[0]) ] )
			result[:2] = f1 - f2
			constrain[2][1] = 0.5 * self.link_lengths[0]
			constrain[2][2] = 0.5 * self.link_lengths[0] * math.sin(-state[1])
			constrain[2][3] = 0.5 * self.link_lengths[0] * math.cos(-state[1])	## not sure
			constrain[2][12] = (1/12) * self.link_masses[0]*pow(self.link_lengths[0], 2)
			result[2] = action[0] - action[1] - self.joint_viscous_friction * q0dot
			constrain[3:5, 6:8] = [[1,0],[0,1]]
			constrain[5][12] = -1 
			constrain[5][15] = 1 	
			
			w1 = state[3] + state[self.num_links + 1] # not sure
			q1dot = state[self.num_links + 1]
			constrain[6:8, 2:4] = [[1,0],[0,1]]
			constrain[6:8, 4:6] = [ [-math.cos(-state[2]), math.sin(-state[2]) ], [ -math.sin(-state[2]), -math.cos(-state[2]) ] ]
			constrain[6:8, 8:10] = [ [ -self.link_masses[1], 0 ],[ 0, -self.link_masses[1] ] ]	
			constrain[7][13] = -self.link_masses[1] * 0.5 * self.link_lengths[1]
			f1 = np.array( [ -self.link_masses[1]*0.5*self.link_lengths[1]*pow(w1, 2),  0 ] )
			f2 = np.array( [ self.link_masses[1] * 9.8 * math.sin(state[0] + state[1]), -self.link_masses[1] * 9.8 * math.cos(state[0] + state[1] ) ] )
			result[6:8] =  f1 - f2
			constrain[8][3] = 0.5 * self.link_lengths[1]
			constrain[8][4] = 0.5 * self.link_lengths[1] * math.sin(-state[2])
			constrain[8][5] = 0.5 * self.link_lengths[1] * math.cos(-state[2])
			constrain[8][13] = (1/12) * self.link_masses[1]*pow(self.link_lengths[1], 2)
			result[8] = action[1] - action[2] - self.joint_viscous_friction * q1dot
			R = [ [ math.cos(-state[1]), -math.sin(-state[1]) ], [ math.sin(-state[1]), math.cos(-state[1]) ] ]
			Rt = np.linalg.inv(R) 	
			#print(R)
			#print(Rt)
			constrain[9:11,6:8] = Rt
			trans = np.dot(Rt * self.link_lengths[0], np.array([[0],[1]]))
			#print(trans)
			constrain[9:11,8:10] = [[-1,0],[0,-1]]
			constrain[9:11,12:13] = trans
			trans_res = np.dot(Rt * self.link_lengths[0] * pow(w0, 2), np.array( [ [-1] , [0] ] ) )
			result[9:11] = np.transpose(trans_res)
			#print(trans_res)
			#print(result)
			constrain[11][12] = 1 
			constrain[11][16] = 1 
			constrain[11][13] = -1
			
			w2 = state[3] + state[4] + state[5] # not sure
			q2dot = state[5]						
			constrain[12:14, 4:6] = [[1,0],[0,1]]
			constrain[12:14, 10:12] = [ [ -self.link_masses[2], 0 ],[ 0, -self.link_masses[2] ] ]			
			constrain[13][14] = -self.link_masses[2] * 0.5 * self.link_lengths[2]
			f1 = np.array( [ -self.link_masses[2]*0.5*self.link_lengths[2]*pow(w2, 2),  0 ] )
			f2 = np.array( [ self.link_masses[2] * 9.8 * math.sin(state[0] + state[1] + state[2]), -self.link_masses[2] * 9.8 * math.cos(state[0] + state[1] +state[2] ) ] )
			result[12:14] =  f1 - f2
			constrain[14][5] = 0.5 * self.link_lengths[2]
			constrain[14][14] = (1/12) * self.link_masses[2]*pow(self.link_lengths[2], 2)
			result[14] = action[2] - self.joint_viscous_friction * q2dot 
			R = [ [ math.cos(-state[2]), -math.sin(-state[2]) ], [ math.sin(-state[2]), math.cos(-state[2]) ] ]
			Rt = np.linalg.inv(R) 			
			constrain[15:17,8:10] = Rt
			trans = np.dot(Rt * self.link_lengths[1], np.array([[0],[1]]))
			#print(trans)
			constrain[15:17,10:12] = [[-1,0],[0,-1]]
			constrain[15:17,13:14] = trans
			trans_res = np.dot(Rt * self.link_lengths[1] * pow(w1, 2), np.array( [ [-1] , [0] ] ) )
			result[15:17] = np.transpose(trans_res)			
			constrain[17][13] = 1 
			constrain[17][17] = 1 
			constrain[17][14] = -1						
			
			
			result = np.transpose(result)		
			solution = np.linalg.solve(constrain, result)
			q0dd = solution[-3]
			q1dd = solution[-2]
			q2dd = solution[-1]
			state[3] += q0dd * dt
			state[0] += state[3]*dt + 0.5 * q0dd *dt *dt
			state[4] += q1dd * dt
			state[1] += state[4]*dt + 0.5 * q1dd * dt * dt
			state[5] += q2dd * dt
			state[2] += state[5]*dt + 0.5 * q2dd * dt * dt			
			
			return state 
								
		for i in range(self.num_links):

			if i == 0:
				if self.num_links == 1:
					w = state[1]
					qdot = state[1]				
					constrain[:2, :2] = [[1,0],[0,1]]
					constrain[:2,2:4] = [ [ -self.link_masses[i], 0 ],[ 0, -self.link_masses[i] ] ]
					constrain[1][4] = -self.link_masses[i] * 0.5 * self.link_lengths[i]
					f1 = np.array( [ -self.link_masses[i]*0.5*self.link_lengths[i]*pow(w, 2),  0 ] )
					f2 = np.array( [ self.link_masses[i] * 9.8 * math.sin(state[0]), -self.link_masses[i] * 9.8 * math.cos(state[0]) ] )
					result[:2] = f1 - f2  
					constrain[2][1] = 0.5 * self.link_lengths[i]
					constrain[2][4] = (1/12) * self.link_masses[i]*pow(self.link_lengths[i], 2) 
					result[2] = action[i] - self.joint_viscous_friction * qdot
					constrain[3:5, 2:4] = [[1,0],[0,1]]
					constrain[5][4:] = [-1,1]
					
					
				else:
					w0 = state[self.num_links]
					q0dot = state[self.num_links]
					#print(math.sin(state[0]), math.cos(state[0]))
					constrain[:2, :2] = [[1,0],[0,1]]
					constrain[:2,2:4] = [ [-math.cos(-state[1]), math.sin(-state[1]) ], [ -math.sin(-state[1]), -math.cos(-state[1]) ] ]
					constrain[:2,4:6] = [ [ -self.link_masses[i], 0 ],[ 0, -self.link_masses[i] ] ]
					constrain[1][8] = -self.link_masses[i] * 0.5 * self.link_lengths[i]
					f1 = np.array( [ -self.link_masses[i]*0.5*self.link_lengths[i]*pow(w0, 2),  0 ] )
					f2 = np.array( [ self.link_masses[i] * 9.8 * math.sin(state[0]), -self.link_masses[i] * 9.8 * math.cos(state[0]) ] )
					result[:2] = f1 - f2
					constrain[2][1] = 0.5 * self.link_lengths[i]
					constrain[2][2] = 0.5 * self.link_lengths[i] * math.sin(-state[1])
					constrain[2][3] = 0.5 * self.link_lengths[i] * math.cos(-state[1])	## not sure
					constrain[2][8] = (1/12) * self.link_masses[i]*pow(self.link_lengths[i], 2)
					result[2] = action[i] - action[i+1] - self.joint_viscous_friction * q0dot
					constrain[3:5, 4:6] = [[1,0],[0,1]]
					constrain[5][8] = -1 
					constrain[5][10] = 1 
						 					
			elif i == self.num_links - 1:
				w = state[2] + state[self.num_links + 1] # not sure
				qdot = state[self.num_links + 1]
				constrain[6:8, 2:4] = [[1,0],[0,1]]
				constrain[6:8, 6:8] = [ [ -self.link_masses[i], 0 ],[ 0, -self.link_masses[i] ] ]
				constrain[7][9] = -self.link_masses[i] * 0.5 * self.link_lengths[i]
				f1 = np.array( [ -self.link_masses[i]*0.5*self.link_lengths[i]*pow(w, 2),  0 ] )
				f2 = np.array( [ self.link_masses[i] * 9.8 * math.sin(state[0] + state[1]), -self.link_masses[i] * 9.8 * math.cos(state[0] + state[1] ) ] )				
				result[6:8] =  f1 - f2
				constrain[8][3] = 0.5 * self.link_lengths[i]
				constrain[8][9] = (1/12) * self.link_masses[i]*pow(self.link_lengths[i], 2)
				result[8] = action[i] - self.joint_viscous_friction * qdot  
				#print(action[i], qdot)
				#print(result)
				R = [ [ math.cos(-state[i]), -math.sin(-state[i]) ], [ math.sin(-state[i]), math.cos(-state[i]) ] ]
				Rt = np.linalg.inv(R) 	
				#print(R)
				#print(Rt)
				constrain[9:11,4:6] = Rt
				trans = np.dot(Rt * self.link_lengths[i-1], np.array([[0],[1]]))
				#print(trans)
				constrain[9:11,6:8] = [[-1,0],[0,-1]]
				constrain[9:11,8:9] = trans
				trans_res = np.dot(Rt * self.link_lengths[i-1] * pow(state[i-1], 2), np.array( [ [-1] , [0] ] ) )
				result[9:11] = np.transpose(trans_res)
				#print(trans_res)
				#print(result)
				constrain[11][8] = 1 
				constrain[11][11] = 1 
				constrain[11][9] = -1
				 	
		result = np.transpose(result)		
		solution = np.linalg.solve(constrain, result)
		
		if self.num_links == 1:	
			qdd = solution[-1]
			state[1] += qdd * dt
			state[0] += state[1]*dt + 0.5*qdd*dt*dt
			return state
		#print(solution)
		q0dd = solution[-2]
		q1dd = solution[-1]
		state[2] += q0dd * dt
		state[0] += state[2]*dt + 0.5 * q0dd *dt *dt
		state[3] += q1dd * dt
		state[1] += state[3]*dt + 0.5 * q1dd * dt * dt
        
        # Replace this with your code:
		return state

