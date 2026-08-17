#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dog_ai_detection.parking_detection_node
=======================================
节点 1: 车辆违规停放检测 (行人作为测试类别).

检测流程 (单模型一次推理):
    yolo26n (COCO 80 类)
      ├── parking 组: car/truck/bus/motorcycle -> 禁停区域多边形判定 -> 违规红框
      └── pedestrian 组: person (测试用, 蓝框, 不判违规)

话题 (标准消息, 无自定义消息):
    订阅: /camera/color/image_raw, /camera/depth/image_raw, /camera/color/camera_info
    发布: /detection/result_image, /detection/object_markers,
          /detection/map_markers, /detection/alert, /detection/web_json

启动: 见 dog_bringup/scripts/start_detection.sh
"""

import rclpy

from dog_ai_detection.detection_core import BaseVizDetectionNode


class ParkingDetectionNode(BaseVizDetectionNode):
    """车辆违规停放 + 行人(测试) 检测节点."""

    TASK_NODE_NAME = 'parking_detection_node'
    MODEL_CANDIDATES = ('yolo26n.engine', 'yolo26n.pt')

    def _declare_task_params(self):
        self.declare_parameter('result_image_topic', '/detection/result_image')
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
