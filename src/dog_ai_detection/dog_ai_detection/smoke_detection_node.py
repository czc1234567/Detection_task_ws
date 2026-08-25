#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点 4: 吸烟检测."""

import rclpy
from rclpy.executors import ExternalShutdownException
from dog_ai_detection.detection_core import BaseVizDetectionNode


class SmokeDetectionNode(BaseVizDetectionNode):
    TASK_NODE_NAME = 'smoke_detection_node'
    MODEL_CANDIDATES = ('smoking.engine', 'smoking.pt')

    def _declare_task_params(self):
        self.declare_parameter('smoke_classes', ['smoke', 'smoking', 'cigarette'])

    def build_groups(self):
        smoke_cls = list(self.get_parameter('smoke_classes').value)
        return [
            {
                'name': 'smoke',
                'classes': smoke_cls,
                'offset': 4000,
                'zones': False,
                'violation_classes': tuple(smoke_cls),
                'color_ok': (0.8, 0.5, 0.2),
                'color_vio': (1.0, 0.0, 0.0),
                'color_ok_bgr': (50, 150, 255),
                'color_vio_bgr': (0, 0, 255),
            }
        ]


def main(args=None):
    rclpy.init(args=args)
    node = SmokeDetectionNode()
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