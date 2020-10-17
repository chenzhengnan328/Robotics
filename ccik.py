#!/usr/bin/env python3

import math
import numpy
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from cartesian_control.msg import CartesianCommand
from urdf_parser_py.urdf import URDF
import random
import tf
import time
from threading import Thread, Lock

'''This is a class which will perform both cartesian control and inverse
   kinematics'''
class CCIK(object):
    def __init__(self):
    #Load robot from parameter server
        self.robot = URDF.from_parameter_server()

    #Subscribe to current joint state of the robot
        rospy.Subscriber('/joint_states', JointState, self.get_joint_state)

    #This will load information about the joints of the robot
        self.num_joints = 0
        self.joint_names = []
        self.q_current = []
        self.joint_axes = []
        self.get_joint_info()

    #This is a mutex
        self.mutex = Lock()

    #Subscribers and publishers for for cartesian control
        rospy.Subscriber('/cartesian_command', CartesianCommand, self.get_cartesian_command)
        self.velocity_pub = rospy.Publisher('/joint_velocities', JointState, queue_size=10)
        self.joint_velocity_msg = JointState()

        #Subscribers and publishers for numerical IK
        rospy.Subscriber('/ik_command', Transform, self.get_ik_command)
        self.joint_command_pub = rospy.Publisher('/joint_command', JointState, queue_size=10)
        self.joint_command_msg = JointState()

    '''This is a function which will collect information about the robot which
       has been loaded from the parameter server. It will populate the variables
       self.num_joints (the number of joints), self.joint_names and
       self.joint_axes (the axes around which the joints rotate)'''
    def get_joint_info(self):
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link

    '''This is the callback which will be executed when the cartesian control
       recieves a new command. The command will contain information about the
       secondary objective and the target q0. At the end of this callback, 
       you should publish to the /joint_velocities topic.'''
    
    def get_qdot(self, b_T_eed, b_T_eec, joint_transforms):

        eec_T_b = tf.transformations.inverse_matrix(b_T_eec)
        eec_T_eed = numpy.dot(eec_T_b, b_T_eed)
        
        deltax = tf.transformations.translation_from_matrix(eec_T_eed)
        angle, axis = self.rotation_from_matrix(eec_T_eed)
        deltar = numpy.dot(angle, axis)
        Deltax = numpy.hstack((deltax, deltar))
        Deltax = numpy.transpose(Deltax)
        ee_V_ee = Deltax 
        J = self.get_jacobian(b_T_eec, joint_transforms)
        Jplus= numpy.linalg.pinv(J, 0.01)
        Jminus = numpy.linalg.pinv(J)
        
        qdot = numpy.dot(Jplus, ee_V_ee)
    	
        return qdot, Jminus, J
    
    
    
    def get_desire_position(self, group):
        desire_rotation_component , desire_tranlation_component = [], []
        desire_rotation_component.append(group.rotation.x)
        desire_rotation_component.append(group.rotation.y)
        desire_rotation_component.append(group.rotation.z)
        desire_rotation_component.append(group.rotation.w)
        desire_rotation = tf.transformations.quaternion_matrix(desire_rotation_component)
    	
        
        desire_tranlation_component.append(group.translation.x)
        desire_tranlation_component.append(group.translation.y)
        desire_tranlation_component.append(group.translation.z)
        desire_translation = tf.transformations.translation_matrix(desire_tranlation_component)
        
        b_T_eed = numpy.dot(desire_translation,desire_rotation)
    
        return b_T_eed
    
    
    
    
    def get_cartesian_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR CARTESIAN CONTROL HERE
    
        b_T_eed = self.get_desire_position(command.x_target)
        q0_desire = command.q0_target
        joint_transforms, b_T_eec = self.forward_kinematics(self.q_current)
	
        qdot, Jminus, J = self.get_qdot(b_T_eed, b_T_eec, joint_transforms)
	

        if command.secondary_objective:
            qsec = numpy.zeros(self.num_joints)
            p_gain = 3 
            qsec[0] = p_gain * (q0_desire - self.q_current[0])
            qsec = numpy.transpose(qsec)
            identity_matrix = numpy.identity(self.num_joints)
            null_projection = numpy.dot(Jminus, J)
            null = (identity_matrix - null_projection)
            q_on_null = numpy.dot(null, qsec)
            qdot += q_on_null
        
        
        self.joint_velocity_msg.name = self.joint_names
        self.joint_velocity_msg.velocity = qdot
        self.velocity_pub.publish(self.joint_velocity_msg)
        
        
        

        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This is a function which will assemble the jacobian of the robot using the
       current joint transforms and the transform from the base to the end
       effector (b_T_ee). Both the cartesian control callback and the
       inverse kinematics callback will make use of this function.
       Usage: J = self.get_jacobian(b_T_ee, joint_transforms)'''
    def get_jacobian(self, b_T_eec, joint_transforms):
        J = numpy.zeros((6,self.num_joints))
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR ASSEMBLING THE CURRENT JACOBIAN HERE
        
        for j in range(self.num_joints):
            b_T_j = joint_transforms[j]
            j_T_b = tf.transformations.inverse_matrix(joint_transforms[j])
            j_T_eec = numpy.dot(j_T_b, b_T_eec)
            j_translation_eec = tf.transformations.translation_from_matrix(j_T_eec)
            eec_T_j = tf.transformations.inverse_matrix(j_T_eec)
            #eec_rotation_j = tf.transformations.rotation_from_matrix(j_T_eec)
            #eec_rotation_j = tf.transformations.quaternion_from_matrix(j_T_eec)
            eec_rotation_j = eec_T_j[:3, :3]
            
       
            x, y, z = j_translation_eec[0], j_translation_eec[1], j_translation_eec[2]
            s = numpy.array([[0, -z, y],[z, 0 ,-x],[-y, x, 0]])
       
            Sj_T_eec = numpy.dot(-eec_rotation_j , s)
            #Vj = numpy.hstack((Sj_T_eec, eec_rotation_j))
            #Vj = numpy.transpose(Vj)
            #Vj.append(Sj_T_eec)
            #Vj = np.array(Vj)
            Vj = numpy.vstack((Sj_T_eec, eec_rotation_j))
            Jj = numpy.dot(Vj, self.joint_axes [j])
            
            J[:, j] = Jj 
            
            
            


        #--------------------------------------------------------------------------
        return J

    '''This is the callback which will be executed when the inverse kinematics
       recieve a new command. The command will contain information about desired
       end effector pose relative to the root of your robot. At the end of this
       callback, you should publish to the /joint_command topic. This should not
       search for a solution indefinitely - there should be a time limit. When
       searching for two matrices which are the same, we expect numerical
       precision of 10e-3.'''
    def get_ik_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR INVERSE KINEMATICS HERE

        b_T_eed = self.get_desire_position(command)       
        delta_q = numpy.ones(self.num_joints)
        marker = False
        
        for _ in range(3):
            q_current = []
            
            for _ in range(self.num_joints):
                q_current.append(random.random())
                
            q_current = numpy.transpose(q_current)
               
            start_point = time.time()
            
            while time.time() - start_point < 10 and not all( abs(deviation) < 0.001 for deviation in delta_q):
           
                joint_transforms, b_T_eec = self.forward_kinematics(q_current)
                
                delta_q, _ ,_ = self.get_qdot(b_T_eed, b_T_eec, joint_transforms)
                
                q_current += delta_q
           
           
            if all( abs(deviation) < 0.001 for deviation in delta_q):
                marker = True
                break
        
        if marker == True:
            self.joint_command_msg.name = self.joint_names
            self.joint_command_msg.position = q_current
            self.joint_command_pub.publish(self.joint_command_msg)
        else:
            print('IK Fails')
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This function will return the angle-axis representation of the rotation
       contained in the input matrix. Use like this: 
       angle, axis = rotation_from_matrix(R)'''
    def rotation_from_matrix(self, matrix):
        R = numpy.array(matrix, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = numpy.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = numpy.linalg.eig(R)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis

    '''This is the function which will perform forward kinematics for your 
       cartesian control and inverse kinematics functions. It takes as input
       joint values for the robot and will return an array of 4x4 transforms
       from the base to each joint of the robot, as well as the transform from
       the base to the end effector.
       Usage: joint_transforms, b_T_ee = self.forward_kinematics(joint_values)'''
    def forward_kinematics(self, joint_values):
        joint_transforms = []

        link = self.robot.get_root()
        T = tf.transformations.identity_matrix()

        while True:
            if link not in self.robot.child_map:
                break

            (joint_name, next_link) = self.robot.child_map[link][0]
            joint = self.robot.joint_map[joint_name]

            T_l = numpy.dot(tf.transformations.translation_matrix(joint.origin.xyz), tf.transformations.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2]))
            T = numpy.dot(T, T_l)

            if joint.type != "fixed":
                joint_transforms.append(T)
                q_index = self.joint_names.index(joint_name)
                T_j = tf.transformations.rotation_matrix(joint_values[q_index], numpy.asarray(joint.axis))
                T = numpy.dot(T, T_j)

            link = next_link
        return joint_transforms, T #where T = b_T_ee

    '''This is the callback which will recieve and store the current robot
       joint states.'''
    def get_joint_state(self, msg):
        self.mutex.acquire()
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])
        self.mutex.release()


if __name__ == '__main__':
    rospy.init_node('cartesian_control_and_IK', anonymous=True)
    CCIK()
    rospy.spin()
