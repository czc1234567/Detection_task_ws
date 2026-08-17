#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dog_ai_detection.helmet_vest_detection_node
===========================================
节点 2: 安全帽 / 安全服检测节点.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from dog_ai_detection.detection_core import BaseVizDetectionNode


class HelmetVestDetectionNode(BaseVizDetectionNode):
    """安全帽 / 安全服检测节点."""

    TASK_NODE_NAME = 'helmet_vest_detection_node'
    MODEL_CANDIDATES = ('helmet.engine', 'helmet.pt')

    def _declare_task_params(self):
        self.declare_parameter('result_image_topic', '/detection/ppe_result_image')
        self.declare_parameter('ppe_classes', ['Helm', 'No_Helm', 'No_Vest', 'Vest'])

    def build_groups(self):
        self.ppe_classes = list(self.get_parameter('ppe_classes').value)
        return [
            {'name': 'ppe', 'classes': self.ppe_classes,
             'offset': 2000, 'zones': False,
             'violation_classes': ('No_Helm', 'No_Vest'),
             'color_ok': (1.0, 0.6, 0.0), 'color_vio': (1.0, 0.0, 0.0),
             'color_ok_bgr': (0, 165, 255), 'color_vio_bgr': (0, 0, 255)},
        ]


def main(args=None):
    rclpy.init(args=args)
    node = HelmetVestDetectionNode()
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