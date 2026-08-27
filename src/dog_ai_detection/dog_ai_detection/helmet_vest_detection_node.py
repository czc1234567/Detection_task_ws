#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点 2: 安全帽 / 安全服检测节点."""

import sys
import time
import signal
import threading
import numpy as np
import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from dog_ai_detection.detection_core import BaseVizDetectionNode


class HelmetVestDetectionNode(BaseVizDetectionNode):
    TASK_NODE_NAME = 'helmet_vest_detection_node'
    MODEL_CANDIDATES = ('helmet.engine', 'helmet.pt')

    def _declare_task_params(self):
        self.declare_parameter('ppe_classes', ['Helm', 'No_Helm', 'No_Vest', 'Vest'])

    def build_groups(self):
        self.ppe_classes = list(self.get_parameter('ppe_classes').value)
        # 显式对齐类别映射
        self.class_names = self.ppe_classes
        return [
            {'name': 'ppe', 'classes': self.ppe_classes,
             'offset': 2000, 'zones': True,
             'violation_classes': ('No_Helm', 'No_Vest'),
             'color_ok': (1.0, 0.6, 0.0), 'color_vio': (1.0, 0.0, 0.0),
             'color_ok_bgr': (0, 165, 255), 'color_vio_bgr': (0, 0, 255)},
        ]


def main(args=None):
    rclpy.init(args=args)
    node = HelmetVestDetectionNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    running = True
    def stop_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop_signal)

    win_name = f"Detection - {node.get_name()}"
    if node.show_gui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 640, 360)
        cv2.moveWindow(win_name, 720, 50)  # 右上角
        
        init_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(init_frame, f"Waiting for {node.get_name()}...", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(win_name, init_frame)
        cv2.waitKey(1)

    try:
        while running and rclpy.ok():
            show_frame = None
            with node._frame_lock:
                if node.display_frame is not None:
                    show_frame = node.display_frame

            if node.show_gui:
                if show_frame is not None:
                    cv2.imshow(win_name, show_frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        executor.shutdown()
        if node.show_gui:
            cv2.destroyWindow(win_name)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()