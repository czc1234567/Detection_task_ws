#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点 3: 火焰检测."""

import rclpy
from rclpy.executors import ExternalShutdownException
from dog_ai_detection.detection_core import BaseVizDetectionNode


class FireDetectionNode(BaseVizDetectionNode):
    TASK_NODE_NAME = 'fire_detection_node'
    MODEL_CANDIDATES = ('fire_detector.engine', 'fire_detector.pt')

    def _declare_task_params(self):
        self.declare_parameter('fire_classes', ['fire', 'flame'])

    def build_groups(self):
        fire_cls = list(self.get_parameter('fire_classes').value)
        return [
            {
                'name': 'fire',
                'classes': fire_cls,
                'offset': 3000,
                'zones': False,
                'violation_classes': tuple(fire_cls),
                'color_ok': (1.0, 0.0, 0.0),
                'color_vio': (1.0, 0.0, 0.0),
                'color_ok_bgr': (0, 0, 255),
                'color_vio_bgr': (0, 0, 255),
            }
        ]


def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionNode()
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