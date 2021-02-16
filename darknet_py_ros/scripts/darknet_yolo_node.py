#!/usr/bin/env python3

# Using threading to run node
# See: https://github.com/akio/mask_rcnn_ros/blob/kinetic-devel/nodes/mask_rcnn_node

import os
import sys
import cv2
import numpy as np
import time
import random
import threading
import copy

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
      
      self._last_msg = None
      self._msg_lock = threading.Lock()
      self._publish_rate = rospy.get_param('~publish_rate', 100)
      self.frame = 0
      self.tracker = cv2.TrackerKCF_create()
      self.tracker_is_init = False
      
    def run(self):
      # TODO: change this to generic image and topics later
      # for debug
      self.image_sub = rospy.Subscriber("/zedm/zed_node/rgb/image_rect_color", Image, self._iamge_callback,
                                        queue_size=1, buff_size=2**24)
      self.bb_pub = rospy.Publisher('darknet_py_ros/bounding_boxes',BoundingBoxes, queue_size=1)
      self.object_count_pub = rospy.Publisher('darknet_py_ros/object_count', ObjectCount, queue_size=1)
      self.detection_img_pub = rospy.Publisher('darknet_py_ros/detection_img',
                                    Image, queue_size=1) 
    
      rate = rospy.Rate(self._publish_rate)
      while not rospy.is_shutdown():
        if self._msg_lock.acquire(False):
          msg = self._last_msg
          self._last_msg = None
          self._msg_lock.release()
        else:
          rate.sleep()
          continue
        
        if msg is not None:
          try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
          except CvBridgeError as e:
            rospy.logerr(f"Error getting image: {e}")

          frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
          # TODO: change width and height to equal network width and height
          frame_resized = cv2.resize(frame_rgb, (320, 320),
                                      interpolation=cv2.INTER_LINEAR)
          
          frame_detection = copy.deepcopy(frame_resized)

          img_for_detect = dn.make_image(320, 320, 3)
          dn.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
                    
          self.frame += 1
          if self.frame % 30 == 0 or self.frame == 1 or not self.tracker_is_init:
            # run inference
            prev_time = time.time()
            detections = dn.detect_image(self.network, self.class_names, img_for_detect, thresh=self.thresh)
            rospy.loginfo(f"nUmber of detections: {len(detections)}")

            fps = int(1/(time.time() - prev_time))
            print("FPS: {}".format(fps))

            dn.print_detections(detections, True)

            if detections:
              bbox = [int(val) for val in detections[0][2]]
              del self.tracker
              self.tracker = cv2.TrackerKCF_create()
              self.tracker.init(frame_detection, bbox)
              self.tracker_is_init = True
            else:
              self.tracker_is_init = False

            dn.free_image(img_for_detect)           
          else:
            # tracker is KCF
            ret, roi = self.tracker.update(frame_detection)
            rospy.logdebug("Running tracker")
            if ret:
              detection = ('Drone', 99.99, roi)
              detections = (detection,)
            
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
            image_out.header = msg.header
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
            # bounding_boxes.header.stamp = msg.header.stamp
            bounding_boxes.header.frame_id = "detection"
            bounding_boxes.num_boxes = num_detections
            bounding_boxes.image_header = msg.header
            self.bb_pub.publish(bounding_boxes)

          else:
            # publish 0 object count
            object_count_msg = ObjectCount()
            object_count_msg.header.stamp = rospy.Time.now() # equivalent to rospy.get_rostime()
            object_count_msg.header.frame_id = "detection"
            object_count_msg.count = 0
            self.object_count_pub.publish(object_count_msg)
  
    def _iamge_callback(self, msg):
      """
      callback to receive and process images
      """
      rospy.logdebug("Got an image")
      if self._msg_lock.acquire(False):
        self._last_msg = msg
        self._msg_lock.release()
      

def main(args):
  """
  Create Node and run node
  """

  rospy.init_node('yolo_dector_node')
  yolo = Yolo()
  yolo.run()


if __name__=='__main__':
  main(sys.argv)