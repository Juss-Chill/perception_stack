#!/usr/bin/env python3
'''
TODO:
Read the images and Point cloud data from the rosbag and perform the predictions using various deep learning networks
'''

'''
Aug 2:
Project the lidar pointcloud onto the camera image every frame and verify the results
'''

import rospy
import cv2
import rosbag
from std_msgs.msg import String
from ultralytics import YOLO

import time

def bag_read():
    # Initialize the ROS node with the name 'talker'
    rospy.init_node('object_detection', anonymous=True)
    
    # Create a publisher object
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    # specify rosbag
    try:
        bag = rosbag.Bag("/home/asl/Muni/workspace/fused_l_cam_jul9.bag")
    except:
        print("Bag not found, Please check the existence of the bag......exiting........")
        exit(-1)
    i = 0.
    # read data from the ROSBag
    for topic, msg, ts in bag.read_messages(topics = []):
        print(topic, ts)
        i += 1
        if(i%4 == 0):
            # process one frame and sleep
            time.sleep(2)
            print("===================================================================")
        if (i==4*20):# process one frame and sleep):
            break

        
        

if __name__ == '__main__':
    try:
        bag_read()
    except rospy.ROSInterruptException:
        pass