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
import cProfile


def draw_bbox(input_img, detections):
    """
        input : Raw Image, YOLO Detection results
        outupt : Rawimage with boundning boxes resulted from the YOLO detections
        Info :
            This function takes the input as detections of YOLO and Raw image, draws bounding boxes onto the image and returns it
    """
    for result in detections:
        # print(result)
        boxes = result.boxes.cpu().numpy()      # stores the detection summary
        classes = result.names                  # stores the dictionary of classes
        # print("BOXES : ", boxes)
        # print("CLASSES : ", classes)

        for box in boxes:
            # print("\n\n\nBOX : ", box)
            # print("BOx XYXY : ", type(box.xyxy), box.xyxy.shape)
            x1, y1, x2, y2 = box.xyxy.reshape(-1) # rehape array of type (1,4) to (4,)
            # print(f"Co ordinates : {x1}, {y1}, {x2}, {y2}")
            # print(type(x1), type(x2), type(y1), type(y2))
            confidence = box.conf.item()
            # print("Confidence : ", confidence)
            class_id = box.cls.item()
            label = f"{classes[class_id]} {confidence:.2f}"

                # Draw bounding box and label
            color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            x1, y1, x2, y2 = int(x1), int(y1), int(x2),int(y2)

            # draw a rectangle and visuzlaise it
            cv2.rectangle(input_img, ( x1, y1 ), ( x2, y2 ), color, 1)
            
            cv2.putText(input_img, label, (x1, y1 - 10), font, 0.5, color, thickness=1, lineType=cv2.LINE_AA)

    return input_img



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
    obj_classification_cam_model = YOLO("yolov5nu.pt")
    # ================================================================================================
    i = 0.
    # read data from the ROSBag
    for topic, msg, ts in bag.read_messages(topics = []):
        print(type(msg), topic, ts)
        i += 1

        if (topic == "/right_cam/image_rect_color/compressed"):
            # print(type(msg), "     Right camera image processing...")
            # print(type(msg.data)) # bytes format

            # decode the image using opencv and dispay it
            msg_np = np.frombuffer(msg.data, dtype='uint8')
            # print("Message shape : ", msg_np.shape, "Type : ", type(msg_np))
            
            raw_img = cv2.imdecode(msg_np,cv2.IMREAD_COLOR)
            # print("RWA Message shape : ", raw_img.shape, "Type : ", type(raw_img))

            # cv2.imshow('RAW Image', raw_img)
            # cv2.waitKey(100)  # keep the window for atleast a second to look for images

            # perform classification
            right_cam_results = obj_classification_cam_model(raw_img)
            # for r in right_cam_results:
            #     print(f"Detected {len(r)} objects in image")
            raw_img = draw_bbox(raw_img, right_cam_results)
            
            # Display the image with bounding boxes
            cv2.imshow("YOLO Right Inference", raw_img)
            cv2.waitKey(100)  # Wait for 1 ms for the window to update

            # print(f"Right camera detections : {right_cam_results}")
            
            # cv2.destroyAllWindows()

            # rt_cam_img.publish(msg)
        # break


        if (topic == "/left_cam/image_rect_color/compressed"):
            # decode the image using opencv and dispay it
            msg_np = np.frombuffer(msg.data, dtype='uint8')
            
            raw_img = cv2.imdecode(msg_np,cv2.IMREAD_COLOR)

            # perform classification
            left_cam_results = obj_classification_cam_model(raw_img)
          
            raw_img = draw_bbox(raw_img, left_cam_results)
            
            # Display the image with bounding boxes
            cv2.imshow("YOLO left Inference", raw_img)
            cv2.waitKey(100)  # Wait for 100 ms for the window to update


        if(i%4 == 0):
            # process one frame of all topics and sleep
            time.sleep(0.5)
            print("===================================================================")
        if (i==4*20):# process 20 frames
            break

        if (topic == "/dominant_obj_info"):
            continue

        if (topic == "/velodyne_points"):
            continue


if __name__ == '__main__':
    try:
        bag_read()
        # cProfile.run('bag_read()')
    except rospy.ROSInterruptException:
        pass