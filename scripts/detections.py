#!/usr/bin/env python3
'''
TODO:
Read the images and Point cloud data from the rosbag and perform the predictions using various deep learning networks
'''

'''
Aug 2:
Project the lidar pointcloud onto the camera image every frame and verify the results
'''
# scientific operations
import numpy as np
import pandas as pd

# point cloud processing
import open3d as o3d

#plotting
import matplotlib.pyplot as plt
import matplotlib.image as img

# deep learning library
import torch
from ultralytics import YOLO

#image processing library
import cv2

# ROS libs
import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# Miscelleneaous
import time

def bag_read():
    # Initialize the ROS node with the name 'talker'
    rospy.init_node('object_detection', anonymous=True)
    
    # Create a publisher object
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rt_cam_img = rospy.Publisher('/right_cam_image/compressed', CompressedImage, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    # specify rosbag
    try:
        bag = rosbag.Bag("/home/asl/Muni/workspace/fused_l_cam_jul9.bag")
        '''
            Topics present in the bag
            1. /dominant_obj_info                       ----- custom_msgs/object_info
            2. /left_cam/image_rect_color/compressed    ----- sensor_msgs/CompressedImage
            3. /right_cam/image_rect_color/compressed   ----- sensor_msgs/CompressedImage
            4. /velodyne_points                         ----- sensor_msgs/PointCloud2
        '''
    except:
        print("Bag not found, Please check the existence of the bag......exiting........")
        exit(-1)
    # ================================================================================================
    # define the YOLO model
    obj_classification_cam_model = YOLO("yolov5n.pt")
    # ================================================================================================
    i = 0.
    # read data from the ROSBag
    for topic, msg, ts in bag.read_messages(topics = ['/right_cam/image_rect_color/compressed']):
        print(type(msg), topic, ts)
        i += 1

        if (topic == "/right_cam/image_rect_color/compressed"):
            print(type(msg), "     Right camera image processing...")
            print(type(msg.data)) # bytes format

            # decode the image using opencv and dispay it
            msg_np = np.frombuffer(msg.data, dtype='uint8')
            # print("Message shape : ", msg_np.shape, "Type : ", type(msg_np))
            
            raw_img = cv2.imdecode(msg_np,cv2.IMREAD_COLOR)
            # print("RWA Message shape : ", raw_img.shape, "Type : ", type(raw_img))

            cv2.imshow('RAW Image', raw_img)
            cv2.waitKey(5)  # keep the window for atleast a second to look for images

            # perform classification
            # right_cam_results = obj_classification_cam_model(raw_img)
            # for r in right_cam_results:
            #     print(f"Detected {len(r)} objects in image")

            # print(f"Right camera detections : {right_cam_results}")
            
            # cv2.destroyAllWindows()

            # rt_cam_img.publish(msg)
        # break
'''
        if(i%4 == 0):
            # process one frame of all topics and sleep
            time.sleep(0.5)
            print("===================================================================")
        if (i==4*20):# process one frame and sleep):
            break

'''

        

if __name__ == '__main__':
    try:
        bag_read()
    except rospy.ROSInterruptException:
        pass