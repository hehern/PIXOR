#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import sys
import glob
import argparse
import os
import time
import ctypes

import pyximport
pyximport.install()

import torch
import numpy as np
from model import PIXOR
from postprocess import filter_pred, non_max_suppression
from utils import get_bev, plot_bev

class Detector(object):

    def __init__(self, config, cdll):
        self.config = config
        self.cdll = cdll
        if self.cdll:
            self.LidarLib = ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        print("device = ", self.device)
        self.net = PIXOR(config['geometry'], config['use_bn']).to(self.device)

        self.net.set_decode(True)
        self.net.load_state_dict(torch.load(config['ckpt_name'], map_location=self.device))
        self.net.eval()

        for p in self.net.parameters():
            p.require_grad = False

        print("PIXOR BEV Detector Initialized!")

    def preprocess(self, velo):
        geom = self.config['geometry']
        velo_processed = np.zeros(geom['input_shape'], dtype=np.float32)
        if self.cdll:
            num = velo.shape[0]
            if not velo.flags['C_CONTIGUOUS']:
                velo = np.ascontiguous(velo, dtype=velo.dtype)  # 如果不是C连续的内存，必须强制转换
            velo_ctypes_ptr = velo.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) 
            c_data = ctypes.c_void_p(velo_processed.ctypes.data)
            self.LidarLib.createTopViewMaps_ros(c_data, velo_ctypes_ptr, ctypes.c_int(num))
        else:
            def passthrough(velo):
                q = (geom['W1'] < velo[:, 0]) * (velo[:, 0] < geom['W2']) * \
                    (geom['L1'] < velo[:, 1]) * (velo[:, 1] < geom['L2']) * \
                    (geom['H1'] < velo[:, 2]) * (velo[:, 2] < geom['H2'])
                indices = np.where(q)[0]
                return velo[indices, :]

            velo = passthrough(velo)
            velo_processed = np.zeros(geom['input_shape'], dtype=np.float32)
            intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
            for i in range(velo.shape[0]):
                x = int((velo[i, 1] - geom['L1']) / 0.1)
                y = int((velo[i, 0] - geom['W1']) / 0.1)
                z = int((velo[i, 2] - geom['H1']) / 0.1)
                velo_processed[x, y, z] = 1
                velo_processed[x, y, -1] += velo[i, 3]
                intensity_map_count[x, y] += 1
            velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1], intensity_map_count, \
                                                 where=intensity_map_count != 0)

        velo_processed = torch.from_numpy(velo_processed).permute(2, 0, 1).to(self.device)
        velo_processed.require_grad=False
        return velo_processed

    def postprocess(self, pred):
        cls_pred = pred[..., 0]
        activation = cls_pred > self.config['cls_threshold']
        num_boxes = int(activation.sum())

        if num_boxes == 0:
            print("No bounding box found")
            return [], []

        corners = torch.zeros((num_boxes, 8))
        for i in range(1, 9):
            corners[:, i - 1] = torch.masked_select(pred[i, ...], activation)
        corners = corners.view(-1, 4, 2).numpy()
        scores = torch.masked_select(cls_pred, activation).cpu().numpy()

        # NMS
        selected_ids = non_max_suppression(corners, scores, self.config['nms_iou_threshold'])
        corners = corners[selected_ids]
        scores = scores[selected_ids]

        return corners, scores

    def __call__(self, velo):
        t_start = time.time()
        bev = self.preprocess(velo)
        t_pre = time.time()
        with torch.no_grad():
            pred = self.net(bev.unsqueeze(0)).squeeze_(0)

        t_m = time.time()
        corners, scores = filter_pred(self.config, pred)
        input_np = bev.permute(1, 2, 0).cpu().numpy()

        t_post = time.time()
        pred_bev = get_bev(input_np, corners)

        t_s = [t_pre-t_start, t_m-t_pre, t_post-t_m]
        return t_s, corners, scores, pred_bev


def cross_product(xp, yp, x1, y1, x2, y2):
    return (x1 - xp) * (y2 - yp)-(x2 - xp) * (y1 - yp)

def isPointInRect(xp, yp, corner):
    apbp = cross_product(xp, yp, corner[0,0], corner[0,1], corner[1,0], corner[1,1])
    bpcp = cross_product(xp, yp, corner[1,0], corner[1,1], corner[2,0], corner[2,1])
    cpdp = cross_product(xp, yp, corner[2,0], corner[2,1], corner[3,0], corner[3,1])
    dpap = cross_product(xp, yp, corner[3,0], corner[3,1], corner[0,0], corner[0,1])
    if (apbp >= 0 and bpcp >= 0 and cpdp >= 0 and dpap >= 0) or (apbp <= 0 and bpcp <= 0 and cpdp <= 0 and dpap <= 0):
        return True
    else:
        return False

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]

def velo_callback(msg):
    # time_start = time.time()
    global pixor, max_marker_size_

    # get the raw pcl msg
    pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=(
        "x", "y", "z", "intensity","ring"))
    np_p = np.array(list(pcl_msg), dtype=np.float32)


    # call the model
    time_cost, corners, scores, pred_bev = pixor(np_p)
    print('scores = ', scores)
    # print("corners.shape = ", corners.shape[0])

    # get the point index of each obstacle
    indexs = []
    if corners is not None:
        for i,j in enumerate(np_p):
            for corner_num in range(corners.shape[0]):
                if isPointInRect(j[0], j[1], corners[corner_num]):
                    indexs.append(i)
                    break
    cloud_obstacle = np_p[indexs]
    cloud_surround = np.delete(np_p, indexs, 0)

    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = msg.header.frame_id
    msg_cloud_obstacle = pc2.create_cloud(header=header,fields=_make_point_field(4),points=cloud_obstacle)
    msg_cloud_surround = pc2.create_cloud(header=header,fields=_make_point_field(4),points=cloud_surround)
    pub_obstacle.publish(msg_cloud_obstacle)
    pub_surroundings.publish(msg_cloud_surround)


    # display the boundingbox
    box_array = MarkerArray()
    box_array.markers.clear()
    marker_id = 0
    if corners is not None:
        for corner in corners:
            
            line_strip = Marker()
            line_strip.header.frame_id = msg.header.frame_id
            line_strip.header.stamp = rospy.Time.now()
            line_strip.ns = "test"
            line_strip.type = Marker.LINE_STRIP
            line_strip.scale.x = 0.1
            line_strip.color.r = 1.0
            line_strip.color.a = 1.0

            p1 = Point()
            p2 = Point()
            p3 = Point()
            p4 = Point()
            line_strip.points = [p1,p2,p3,p4,p1]

            p1.x = corner[0,0]
            p1.y = corner[0,1]
            p1.z = 1
            p2.x = corner[1,0]
            p2.y = corner[1,1]
            p2.z = 1
            p3.x = corner[2,0]
            p3.y = corner[2,1]
            p3.z = 1
            p4.x = corner[3,0]
            p4.y = corner[3,1]
            p4.z = 1

            line_strip.id = marker_id
            line_strip.points[0] = p1
            line_strip.points[1] = p2
            line_strip.points[2] = p3
            line_strip.points[3] = p4
            line_strip.points[4] = p1
            box_array.markers.append(line_strip)

            marker_id += 1
    # clear the extra markers
    if marker_id > max_marker_size_:
        max_marker_size_ = marker_id
    for i in range(marker_id, max_marker_size_):
        line_strip = Marker()
        line_strip.header.frame_id = msg.header.frame_id
        line_strip.header.stamp = rospy.Time.now()
        line_strip.ns = "test"
        line_strip.type = Marker.LINE_STRIP
        line_strip.id = i
        p1 = p2 = p3 = p4 = Point()
        line_strip.points = [p1,p2,p3,p4,p1]
        p1.x = p1.y = p1.z = p2.x = p2.y = p2.z = p3.x = p3.y = p3.z = p4.x = p4.y = p4.z = 0
        line_strip.points[0] = p1
        line_strip.points[1] = p2
        line_strip.points[2] = p3
        line_strip.points[3] = p4
        line_strip.points[4] = p1
        box_array.markers.append(line_strip)
    # publish the boundingbox
    if len(box_array.markers) is not 0:
        pub_arr_bbox.publish(box_array)
        box_array.markers.clear()
    # print('box published')

    # time_end = time.time()
    # print('callback_time = ', time_end-time_start)

#  pixor initializer
def pixor_init():
    global pixor
    config = {
      "ckpt_name": "experiments/default/34epoch",
      "use_bn": True,
      "cls_threshold": 0.5,
      "nms_iou_threshold": 0.1,
      "nms_top": 64,
      "geometry": {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.0,
        'grid_size': 0.1,
        'input_shape': [800, 700, 36],
        'label_shape': [200, 175, 7],
        },
    }
    # Initialize Detector
    cdll = True
    pixor = Detector(config, cdll)



if __name__ == '__main__':


    #  initializing pixor
    pixor_init()     #模型初始化

    global max_marker_size_
    max_marker_size_ = 0

    #  code added for using ROS
    rospy.init_node('pixor_ros_node')

    sub_ = rospy.Subscriber("velodyne_points", PointCloud2,
                            velo_callback, queue_size=1)

    pub_obstacle = rospy.Publisher(
        "velodyne_points_obstacle", PointCloud2, queue_size=1)
    pub_surroundings = rospy.Publisher(
        "velodyne_points_sur", PointCloud2, queue_size=1)
    pub_arr_bbox = rospy.Publisher(
        "pixor_arr_bbox", MarkerArray, queue_size=10)
    print("[+] pixor_ros_node has started!")
    rospy.spin()
