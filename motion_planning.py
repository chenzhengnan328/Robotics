#!/usr/bin/env python3

import numpy
import random
import sys

import geometry_msgs.msg
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import tf
import moveit_commander
from urdf_parser_py.urdf import URDF
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math

def convert_to_message(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]        
    return t

class MoveArm(object):

    def __init__(self):

        #Loads the robot model, which contains the robot's kinematics information
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        self.robot = URDF.from_parameter_server()
        self.base = self.robot.get_root()
        self.get_joint_info()
        self.visited = set()
        # Wait for moveit IK service
        rospy.wait_for_service("compute_ik")
        self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
        print("IK service ready")

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
                                                      moveit_msgs.srv.GetStateValidity)
        print("State validity service ready")

        # MoveIt parameter
        robot_moveit = moveit_commander.RobotCommander()
        self.group_name = robot_moveit.get_group_names()[0]

        #Subscribe to topics
        rospy.Subscriber('/joint_states', JointState, self.get_joint_state)
        rospy.Subscriber('/motion_planning_goal', Transform, self.motion_planning)
        self.current_obstacle = "None"
        rospy.Subscriber('/obstacle', String, self.get_obstacle)

        #Set up publisher
        self.pub = rospy.Publisher('/joint_trajectory', JointTrajectory, queue_size=10)

    '''This callback provides you with the current joint positions of the robot 
     in member variable q_current.'''
    def get_joint_state(self, msg):
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])

    '''This callback provides you with the name of the current obstacle which
    exists in the RVIZ environment. Options are "None", "Simple", "Hard",
    or "Super". '''
    def get_obstacle(self, msg):
        self.current_obstacle = msg.data

    '''This is the callback which will implement your RRT motion planning.
    You are welcome to add other functions to this class (i.e. an
    "is_segment_valid" function will likely come in handy multiple times 
    in the motion planning process and it will be easiest to make this a 
    seperate function and then call it from motion planning). You may also
    create trajectory shortcut and trajectory sample functions if you wish, 
    which will also be called from the motion planning function.'''   
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
        
        
        
    def normalize(self, end, start):
        direction = end - start
        norm = self.get_distance(end, start)
        
        return direction / norm
    
    
        
    def get_distance(self, point1, point2):
        return numpy.linalg.norm((point1 - point2))
    
    
        
    def is_segment_valid(self, start, end):
        D = end - start
        L = numpy.linalg.norm(D)
        if L == 0.0:
            return start, True
        num = L/0.025# 0.0001
        num = numpy.ceil(num)
        step = D/num
        delta = 0
        qi = start
        for i in numpy.arange(0, num):#num+1
            qi= qi + step
            delta += 0.025
            if not self.is_state_valid(qi):
                return qi, False
                
            if delta > 0.09:  
                return qi, True    
        

    def findnearst_point(self, target, lst):
        min_distance = float('inf')
        for i in range(len(lst)):
            distance = numpy.linalg.norm((lst[i].q - target))
            if  distance < min_distance:
                min_distance = distance
                min_idx = i
        return min_idx 
        
    def shortcut(self, route1):
        res = [route1[0]]
        for i in range(1, len(route1)):
          idx = len(res) -1 
          if self.is_goal_possible(res[idx], route1[i]):
              continue
          res.append(route1[i-1])
        res.append(route1[-1])
        return res
      
        
    def is_goal_possible(self, start, end):
        D = end - start
        L = numpy.linalg.norm(D)
        if L < 0.08:
            return False

        num = L/0.06
	
        num = numpy.ceil(num)
        step = D/num

        qi = start
        for i in numpy.arange(0, num):#num+1
            qi= qi + step
            if not self.is_state_valid(qi):
                return False

        return True

        
    def backwardbfs(self, points, goal):
        path = [goal]
        q = [points[-1]]
        while q:
            next_step = q.pop()
            if next_step[1] == -1:
                path.append(next_step[0])
                return path
            path.append(next_step[0])
            q.append(points[next_step[1]])
            
    def find_path(self, candidates):
        final_path = [candidates[0]]
        
        for i in range(1, len(candidates)):
            D = candidates[i] - candidates[i-1]
            cur_distance = numpy.linalg.norm(D)
            nums = numpy.ceil(cur_distance / 0.5)
            
            if nums > 1:
                start_p = candidates[i-1]
                step = D / nums
                for _ in numpy.arange(0, nums):
                    start_p = start_p + step
                    final_path.append(start_p)
            final_path.append(candidates[i])
            
        return final_path           
                                 
    def motion_planning(self, ee_goal):
        print("Starting motion planning")
    ########INSERT YOUR RRT MOTION PLANNING HERE##########
        b_T_g = self.get_desire_position(ee_goal)
        q_current = self.q_current.copy()
        q_current = numpy.array(q_current)
        goal = self.IK(b_T_g)
        star = RRTBranch([], q_current)
        marker = False
        tree = []
        backwardtree = []
        forward_check = False
        backward_check = False
        #print('this is current', q_current)
        if goal==[]:
            print('not valid')
            return
             
 
        
        
        #print('this is tree', tree)
        goal = numpy.array(goal)
        #print('this is goal', goal)
        box = RRTBranch([], goal)
        tree.append(star)
        backwardtree.append(box) 
        #print('--------------------------------------------------')
        #print(tree[-1].q, tree[-1].parent)
        #print(backwardtree[-1].q, backwardtree[-1].parent)
        print('---------------------------------------------------')
        while numpy.linalg.norm((tree[-1].q - backwardtree[-1].q)) > 0.01: #
            #forward_check = False
            #backward_check = False                                                             
            point = numpy.zeros(self.num_joints)
            
            for i in range(len(point)):
                point[i] = random.uniform(-math.pi, math.pi)
            #print(point)    
            if not self.is_state_valid(point):
                continue
                
            nearest_foward_idx = self.findnearst_point(point, tree)
            nearest_backward_idx = self.findnearst_point(point, backwardtree)
            
            #print('the Findex is', nearest_foward_idx)  
            #print('the Windex is', nearest_backward_idx)   
            nearest_forward_point = tree[nearest_foward_idx]
            #print(nearest_foward_idx)
            nearest_backward_point = backwardtree[nearest_backward_idx]
            
            
            #print('the nearest_point is', nearest_point)
            forward_direction = point - nearest_forward_point.q
            backward_direction = point - nearest_backward_point.q
            
            forward_length = numpy.linalg.norm((point - nearest_forward_point.q))
            backward_length = numpy.linalg.norm((point - nearest_backward_point.q))
            
            if forward_length >= 0.1:
                forward_end_point = nearest_forward_point.q + (forward_direction/forward_length) * 0.1
            else:
                forward_end_point = nearest_forward_point.q + forward_direction
            forward_check = self.is_goal_possible( nearest_forward_point.q , forward_end_point)
            
            if backward_length >= 0.1:
                backward_end_point = nearest_backward_point.q + (backward_direction / backward_length ) * 0.1
            else:
                backward_end_point = nearest_backward_point.q + backward_direction
            backward_check = self.is_goal_possible( nearest_backward_point.q , backward_end_point)
            
            
            #forward_check = self.is_goal_possible( nearest_forward_point.q , forward_end_point)
            #print('the Findex is', next_forward_point)
            if forward_check:
                
                former = nearest_forward_point.parent.copy()
                former.append(nearest_forward_point.q.copy())
                forward_brench = RRTBranch(former , numpy.array(forward_end_point))
                tree.append(forward_brench)
               
                

            if backward_check:
                #print(numpy.linalg.norm((nearest_backward_point.q - backward_end_point)))
                later = nearest_backward_point.parent.copy()
                #print(len(later))
                later.append(nearest_backward_point.q.copy())
                backward_brench = RRTBranch(later , numpy.array(backward_end_point))
                backwardtree.append(backward_brench)  
                         
            #print('the Findex is', len(backwardtree))
            if self.is_goal_possible(tree[-1].q, backwardtree[-1].q):
                break
        #    print(len(tree) + len(backwardtree))
        print('find route')
        lujing = []        
      
        
        forward_mid = tree[-1].q 
        for i in range(len(tree[-1].parent)):
            lujing.append( tree[-1].parent[i])
        lujing.append(numpy.array(forward_mid))

       
        backward_mid = backwardtree[-1].q 
        lujing.append(numpy.array(backward_mid))
        backwardtree[-1].parent.reverse()
        for i in range(len(backwardtree[-1].parent)):
            lujing.append( backwardtree[-1].parent[i])
        
        #print(' the FW q', tree[-1].q )
        #print(' the FW p', tree[-1].parent )
        print('-----------------------------------------------------')
        #print(  'the BW q',backwardtree[-1].q)      
        #print(  'the BW p',backwardtree[-1].parent)   
        #lujing = tree[-1].parent + tree[-1] + backwardtree[-1] + backwardtree[-1].parent
        #print('--------------------------------------------------------------------')
        
        #print(lujing)
        #print('----------------------------------------------------------------')
        final_path = self.shortcut(lujing)
        #print('this is lujinh',lujing)
        print('publish goal')
        output_path = self.find_path(final_path)
        
        
        #print(output_path)
        #print('ready to move')
        
        trajectory= JointTrajectory()
        trajectory.joint_names = self.joint_names
        for i in range(0, len(output_path)):
            p =JointTrajectoryPoint()
            #print(traject[i])
            p.positions = output_path[i]
            trajectory.points.append(p)

        self.pub.publish(trajectory)        
        
        ######################################################

    """ This function will perform IK for a given transform T of the end-effector.
    It returns a list q[] of values, which are the result positions for the 
    joints of the robot arm, ordered from proximal to distal. If no IK solution 
    is found, it returns an empy list.
    """
    def IK(self, T_goal):
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state.name = self.joint_names
        req.ik_request.robot_state.joint_state.position = numpy.zeros(self.num_joints)
        req.ik_request.robot_state.joint_state.velocity = numpy.zeros(self.num_joints)
        req.ik_request.robot_state.joint_state.effort = numpy.zeros(self.num_joints)
        req.ik_request.robot_state.joint_state.header.stamp = rospy.get_rostime()
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = self.base
        req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rospy.Duration(3.0)
        res = self.ik_service(req)
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = res.solution.joint_state.position
        return q

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


    """ This function checks if a set of joint angles q[] creates a valid state,
    or one that is free of collisions. The values in q[] are assumed to be values
    for the joints of the KUKA arm, ordered from proximal to distal. 
    """
    def is_state_valid(self, q):
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = self.group_name
        req.robot_state = moveit_msgs.msg.RobotState()
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = q
        req.robot_state.joint_state.velocity = numpy.zeros(self.num_joints)
        req.robot_state.joint_state.effort = numpy.zeros(self.num_joints)
        req.robot_state.joint_state.header.stamp = rospy.get_rostime()
        res = self.state_valid_service(req)
        return res.valid


'''This is a class which you can use to keep track of your tree branches.
It is easiest to do this by appending instances of this class to a list 
(your 'tree'). The class has a parent field and a joint position field (q). 
You can initialize a new branch like this:
RRTBranch(parent, q)
Feel free to keep track of your branches in whatever way you want - this
is just one of many options available to you.'''
class RRTBranch(object):
    def __init__(self, before, q):
        self.parent = before
        self.q = q


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm', anonymous=True)
    ma = MoveArm()
    rospy.spin()

