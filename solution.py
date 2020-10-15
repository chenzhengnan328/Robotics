#!/usr/bin/env python3  
import rospy

import numpy

import tf
import tf2_ros
import geometry_msgs.msg

def publish_transforms():
	t1 = geometry_msgs.msg.TransformStamped()
	t1.header.stamp = rospy.Time.now()
	t1.header.frame_id = "base_frame"
	t1.child_frame_id = "object_frame"
	
	q1 = tf.transformations.quaternion_from_euler(0.64, 0.64, 0)
	trans = tf.transformations.translation_matrix((1.5,0.8,0))
	tr = numpy.dot(tf.transformations.quaternion_matrix(q1), trans)
	translation = tf.transformations.translation_from_matrix(tr)
	
	t1.transform.translation.x = translation[0]
	t1.transform.translation.y = translation[1]
	t1.transform.translation.z = translation[2]
	t1.transform.rotation.x = q1[0]
	t1.transform.rotation.y = q1[1]
	t1.transform.rotation.z = q1[2]
	t1.transform.rotation.w = q1[3]
	br.sendTransform(t1)
	
	
	
	t2 = geometry_msgs.msg.TransformStamped()
	t2.header.stamp = rospy.Time.now()
	t2.header.frame_id = "base_frame"
	t2.child_frame_id = "robot_frame"
	
	T_rot_y = tf.transformations.rotation_matrix(1.5, (0,1,0))
	trans_z = tf.transformations.translation_matrix((0,0, -2))
	tr2 = numpy.dot(T_rot_y, trans_z)
	
	translation2 = tf.transformations.translation_from_matrix(tr2)
	t2.transform.translation.x = translation2[0]
	t2.transform.translation.y = translation2[1]
	t2.transform.translation.z = translation2[2]
	
	q2 = tf.transformations.quaternion_from_matrix(tr2)
	t2.transform.rotation.x = q2[0]
	t2.transform.rotation.y = q2[1]
	t2.transform.rotation.z = q2[2]
	t2.transform.rotation.w = q2[3]
	br.sendTransform(t2)
	
	
	t3 = geometry_msgs.msg.TransformStamped()
	t3.header.stamp = rospy.Time.now()
	t3.header.frame_id = "robot_frame"
	t3.child_frame_id = "camera_frame"
	
	trans_3 = tf.transformations.translation_matrix((0.3,0,0.3))
	#tr2_inverse = tf.transformations.inverse_matrix(tr2)
	
	camera_in_base = numpy.dot(tr2, trans_3)
	base_in_camera = tf.transformations.inverse_matrix(camera_in_base)
	trans_object = numpy.append(translation, 1)
	object_in_camera = numpy.dot(base_in_camera[0:3], trans_object)
	camera_axis = numpy.array([1,0,0])
	rotation_axis = numpy.cross(camera_axis, object_in_camera)
	angle = numpy.arccos(numpy.dot(camera_axis, object_in_camera)/(numpy.linalg.norm(camera_axis) *numpy.linalg.norm(rotation_axis))) 
	
	robo_to_camera = tf.transformations.quaternion_about_axis(angle, rotation_axis)
	t3.transform.translation.x = 0.3
	t3.transform.translation.y = 0.0
	t3.transform.translation.z = 0.3
	t3.transform.rotation.x = robo_to_camera[0]
	t3.transform.rotation.y = robo_to_camera[1] 
	t3.transform.rotation.z = robo_to_camera[2]
	t3.transform.rotation.w = robo_to_camera[3]
	br.sendTransform(t3)
	
	
	
if __name__ == '__main__':
    rospy.init_node('solution')

    br = tf2_ros.TransformBroadcaster()
    rospy.sleep(0.1)

    while not rospy.is_shutdown():
        publish_transforms()
        rospy.sleep(0.1)
