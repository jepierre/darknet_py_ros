#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import time
import random

# Darknet Lib
import darknet as dn

# For ROS:
import rospy
import rospkg
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from darknet_py_ros_msgs.msg import BoundingBoxes, ObjectCount, BoundingBox


class Yolo:
    
    def __init__(self):
      self.bridge = CvBridge()
      PATH = os.path.dirname(__file__)
      os.chdir(PATH)
      cfg_file = os.path.join("../data/cfg", "yolo-drone.cfg")
      data_file = os.path.join("../data", "drone.data")
      weights = os.path.join("../data/weights", "yolo-drone_199000.weights")

      cfg_file = rospy.get_param('cfg_file', cfg_file)
      data_file = rospy.get_param('data_file', data_file)
      weights = rospy.get_param('weights', weights)

      self.network, self.class_names, self.class_colors = dn.load_network(
        cfg_file,
        data_file,
        weights,
        batch_size=1
      )

      # threshhold for detection
      self.thresh = .25

      self.width = dn.network_width(self.network)
      self.height = dn.network_height(self.network)
      
      # for debug
      self.image_sub = rospy.Subscriber("/zedm/zed_node/rgb/image_rect_color", Image, self.image_cb,
      # self.image_sub = rospy.Subscriber("image", Image, self.image_cb,
                                        queue_size=1, buff_size=2**24)
      self.bb_pub = rospy.Publisher('darknet_py_ros/bounding_boxes',BoundingBoxes, queue_size=10)
      self.object_count_pub = rospy.Publisher('darknet_py_ros/object_count', ObjectCount, queue_size=1)
      self.detection_img_pub = rospy.Publisher('darknet_py_ros/detection_img',
                                    Image, queue_size=10)
    
    def image_cb(self, data):
      try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

      frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
      frame_resized = cv2.resize(frame_rgb, (self.width, self.height),
                                  interpolation=cv2.INTER_LINEAR)
      img_for_detect = dn.make_image(self.width, self.height, 3)
      dn.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

      # run inference
      prev_time = time.time()
      detections = dn.detect_image(self.network, self.class_names, img_for_detect, thresh=self.thresh)
      rospy.loginfo(f"nUmber of detections: {len(detections)}")

      fps = int(1/(time.time() - prev_time))
      print("FPS: {}".format(fps))

      dn.print_detections(detections, True)
      dn.free_image(img_for_detect)


      if detections:
        # draw boxes
        random.seed(42) # bbox colors
        image_np = dn.draw_boxes(detections, frame_resized, self.class_colors)

        # Let's publish the detection image
        img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_out = Image()
        
        try:
          image_out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
          rospy.logerror(f"Error: {e}")
        image_out.header = data.header
        self.detection_img_pub.publish(image_out)

        
        num_detections = len(detections)
        
        # publish object count
        object_count_msg = ObjectCount()
        object_count_msg.header.stamp = rospy.Time.now() # equivalent to rospy.get_rostime()
        object_count_msg.header.frame_id = "detection"
        object_count_msg.count = num_detections
        self.object_count_pub.publish(object_count_msg)
        
        # publish bounding boxes
        
        bounding_boxes = BoundingBoxes()
        for idx, detection in enumerate(detections):
          rospy.loginfo(f"detection: {detection}")

          # TODO: send image coordinates
          bounding_box = BoundingBox()
          bounding_box.Class = detection[0]
          bounding_box.probability = float(detection[1])
          bounding_box.id = idx
          xmin, ymin, xmax, ymax = dn.bbox2points(detection[2])
          bounding_box.xmin = float(xmin/self.width)
          bounding_box.xmax = float(xmax/self.width)
          bounding_box.ymin = float(ymin/self.height)
          bounding_box.ymax = float(ymax/self.height)

          bounding_boxes.bounding_boxes.append(bounding_box)
          
        bounding_boxes.header.stamp = rospy.Time.now()
        # pass on image timestamp instead
        # bounding_boxes.header.stamp = data.header.stamp
        bounding_boxes.header.frame_id = "detection"
        bounding_boxes.num_boxes = num_detections
        bounding_boxes.image_header = data.header
        self.bb_pub.publish(bounding_boxes)

      else:
        # publish 0 object count
        object_count_msg = ObjectCount()
        object_count_msg.header.stamp = rospy.Time.now() # equivalent to rospy.get_rostime()
        object_count_msg.header.frame_id = "detection"
        object_count_msg.count = 0
        self.object_count_pub.publish(object_count_msg)
      

def main(args):

  rospy.init_node('yolo_dector_node')
  yolo = Yolo()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("ShutDown")
  cv2.destroyAllWindows()

if __name__=='__main__':
  main(sys.argv)