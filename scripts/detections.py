#!/usr/bin/env python3
'''
TODO:
Read the images and Point cloud data from the rosbag and perform the predictions using various deep learning networks
'''

'''
Aug 2:
Project the lidar pointcloud onto the camera image every frame and verify the results 
***(To do this image and Point cloud data cannot be obtained for same frame, so write a node class in python)
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
from cv_bridge import CvBridge

# ROS libs
import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs import point_cloud2


# Miscelleneaous
import time
import cProfile
import yaml
import os

def load_parameters(yaml_file):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

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




def project_pts_on_image(extrinsic_mat, intrinsic_mat, image, points):
    reproj_img = image.copy()

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # Homogeneous coordinates

    # Transform points to camera coordinate system
    points_camera = points_homogeneous @ extrinsic_mat.T

    # Project points onto image plane
    points_camera = points_camera[:, :3]  # Drop homogeneous coordinate
    points_image = (intrinsic_mat @ points_camera.T).T  # Apply intrinsics
    points_image = points_image[:, :2] / points_image[:, 2:3]  # Normalize points

    # Overlay points on the image
    for point in points_image:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < reproj_img.shape[1] and 0 <= y < reproj_img.shape[0]:
            cv2.circle(reproj_img, (x, y), 2, (0, 255, 0), -1)
    
    return reproj_img


def bag_read(lid_to_leftcam_rt, lid_to_rightcam_rt, leftcam_intrin_mat, rightcam_intrin_mat):

    right_cam_img_raw = None
    left_cam_image_raw = None
    pc_points = None
    
    print("FINAL RIGHT : ", lid_to_rightcam_rt, lid_to_rightcam_rt.shape)
    print("FINAL LEFT : ", lid_to_leftcam_rt, lid_to_leftcam_rt.shape)
    print("FINAL RIGHT Int: ", rightcam_intrin_mat, rightcam_intrin_mat.shape)
    print("FINAL LEFT Int: ", leftcam_intrin_mat, leftcam_intrin_mat.shape)

    # Create a publisher object
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rt_cam_img = rospy.Publisher('/right_cam_image_detections',Image , queue_size=10)
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
            right_cam_img_raw = raw_img.copy() # used later to projec the pointcloud
            # print("RWA Message shape : ", raw_img.shape, "Type : ", type(raw_img))

            # cv2.imshow('RAW Image', raw_img)
            # cv2.waitKey(100)  # keep the window for atleast a second to look for images

            # perform classification
            right_cam_results = obj_classification_cam_model(raw_img)
            # for r in right_cam_results:
            #     print(f"Detected {len(r)} objects in image")
            raw_img = draw_bbox(raw_img, right_cam_results)
            
            # Display the image with bounding boxes
            # cv2.imshow("YOLO Right Inference", raw_img)
            # cv2.waitKey(100)  # Wait for 1 ms for the window to update

            # print(f"Right camera detections : {right_cam_results}")
            
            # cv2.destroyAllWindows()
            bridge_img = CvBridge()
            final_rt_image = Image()
            final_rt_image_head = Header()

            final_rt_image_head.frame_id = "map"
            final_rt_image_head.stamp = rospy.Time.now()

            # final_rt_image.data = raw_img
            final_rt_image = bridge_img.cv2_to_imgmsg(raw_img, encoding="passthrough", header=final_rt_image_head)
            rt_cam_img.publish(final_rt_image)
        # break


        if (topic == "/left_cam/image_rect_color/compressed"):
            # decode the image using opencv and dispay it
            msg_np = np.frombuffer(msg.data, dtype='uint8')
            
            raw_img = cv2.imdecode(msg_np,cv2.IMREAD_COLOR)
            left_cam_image_raw = raw_img.copy()

            # perform classification
            left_cam_results = obj_classification_cam_model(raw_img)
          
            raw_img = draw_bbox(raw_img, left_cam_results)
            
            # Display the image with bounding boxes
            # cv2.imshow("YOLO left Inference", raw_img)
            # cv2.waitKey(100)  # Wait for 100 ms for the window to update

        
        if (topic == "/velodyne_points"):

            print("Characteristics of the recieved point cloud msg\n")
            height = msg.height 
            width = msg.width
            is_dense = msg.is_dense

            # print(msg)
            msg_new = point_cloud2.read_points(msg, ['x','y','z'], skip_nans=True)
            # print("Point cloud points : ",type(msg_new))

            data_np = np.asarray(list(msg_new), dtype=float)

            pc = o3d.geometry.PointCloud()
            filtered_pts = data_np[ data_np[:, 1] >= 0] # choose all the points whose Y > 0, That is the camera FOV plane
            pc.points = o3d.utility.Vector3dVector(filtered_pts)
            # print(type(pc.points), pc.points)
            pc_points = filtered_pts # to project on to the image
            # o3d.visualization.draw_geometries([pc])
            
            # print("New camera images:  ", type(right_cam_img_raw), " ,,,,, ", type(left_cam_image_raw))

            print(f"height : {height}, Width : {width}, dense : {is_dense}, data shape : {data_np.shape}")
            


        if(i%4 == 0):
            # cv2.imshow("YOLO Right Inference", right_cam_img_raw)
            # cv2.imshow("YOLO Left Inference", left_cam_image_raw)
            # cv2.waitKey(100)  # Wait for 100 ms for the window to update
            # print("MUNI ",type(pc_points), " , ",pc_points.shape ) 
            # process one frame of all topics and sleep
            rightcam_reproj_image = project_pts_on_image(lid_to_rightcam_rt, rightcam_intrin_mat, right_cam_img_raw, pc_points)
            leftcam_reproj_image = project_pts_on_image(lid_to_leftcam_rt, leftcam_intrin_mat, left_cam_image_raw, pc_points)
            cv2.imshow("YOLO Right Inference", rightcam_reproj_image)
            cv2.imshow("YOLO Left Inference", leftcam_reproj_image)
            cv2.waitKey(100)  # Wait for 100 ms for the 
            
            time.sleep(1.0)
            print("===================================================================")
        if (i==4*20):# process 1 frames
            break

        if (topic == "/dominant_obj_info"):
            continue

        

if __name__ == '__main__':
    try:
         # Initialize the ROS node with the name 'object_detection'
        rospy.init_node('object_detection', anonymous=True)
        # print("Current working direcoty :", os.getcwd())
        # yaml_file = rospy.get_param('~yaml_file', 'src/object_detection/config/projection_matrices.yaml')
        yaml_file = rospy.get_param('~yaml_file', 'projection_matrices.yaml')
        params = load_parameters(yaml_file)

        lid_to_leftcam_rt = params['lidar_to_leftcam']['extrinsics']['rt']
        lid_to_leftcam_rt = np.asarray(lid_to_leftcam_rt, dtype=float)
        
        lid_to_rightcam_rt = params['lidar_to_rightcam']['extrinsics']['rt']
        lid_to_rightcam_rt = np.asarray(lid_to_rightcam_rt, dtype=float)

        leftcam_intrin_mat = params['left_camera']['matrix']
        leftcam_intrin_mat = np.asarray(leftcam_intrin_mat, dtype=float)

        rightcam_intrin_mat = params['right_camera']['matrix']
        rightcam_intrin_mat = np.asarray(rightcam_intrin_mat, dtype=float)
        
        # print("FINAL RIGHT : ", lid_to_rightcam_rt, lid_to_rightcam_rt.shape)
        # print("FINAL LEFT : ", lid_to_leftcam_rt, lid_to_leftcam_rt.shape)
        # print("FINAL RIGHT Int: ", rightcam_intrin_mat, rightcam_intrin_mat.shape)
        # print("FINAL LEFT Int: ", leftcam_intrin_mat, leftcam_intrin_mat.shape)

        bag_read(lid_to_leftcam_rt, lid_to_rightcam_rt, leftcam_intrin_mat, rightcam_intrin_mat)
        # cProfile.run('bag_read()')
    except rospy.ROSInterruptException:
        pass