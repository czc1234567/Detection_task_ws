#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dog_ai_detection.parking_detection_node
=======================================
节点 1: 车辆违规停放检测.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from dog_ai_detection.detection_core import BaseVizDetectionNode


class ParkingDetectionNode(BaseVizDetectionNode):
    """车辆违规停放 + 行人(测试) 检测节点."""

    TASK_NODE_NAME = 'parking_detection_node'
    MODEL_CANDIDATES = ('yolo26n.engine', 'yolo26n.pt')

    def _declare_task_params(self):
        # 注意: 已经移除了对 result_image_topic 的重复声明
        self.declare_parameter('vehicle_classes', ['car', 'truck', 'bus', 'motorcycle'])
        self.declare_parameter('person_classes', ['person'])

    def build_groups(self):
        self.vehicle_classes = list(self.get_parameter('vehicle_classes').value)
        self.person_classes = list(self.get_parameter('person_classes').value)
        return [
            {'name': 'parking', 'classes': self.vehicle_classes,
             'offset': 0, 'zones': True,
             'color_ok': (0.0, 1.0, 0.0), 'color_vio': (1.0, 0.0, 0.0),
             'color_ok_bgr': (0, 255, 0), 'color_vio_bgr': (0, 0, 255)},
            {'name': 'pedestrian', 'classes': self.person_classes,
             'offset': 1000, 'zones': False,
             'color_ok': (0.3, 0.6, 1.0), 'color_vio': (1.0, 0.0, 0.0),
             'color_ok_bgr': (255, 140, 30), 'color_vio_bgr': (0, 0, 255)},
        ]


def main(args=None):
    rclpy.init(args=args)
    node = ParkingDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()