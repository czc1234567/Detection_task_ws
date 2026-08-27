#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点 3: 火焰检测."""

import sys
import time
import signal
import threading
import numpy as np
import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from dog_ai_detection.detection_core import BaseVizDetectionNode


class FireDetectionNode(BaseVizDetectionNode):
    TASK_NODE_NAME = 'fire_detection_node'
    MODEL_CANDIDATES = ('fire_detector.engine', 'fire_detector.pt')

    def _declare_task_params(self):
        self.declare_parameter('fire_classes', ['fire', 'flame'])

    def build_groups(self):
        fire_cls = list(self.get_parameter('fire_classes').value)
        self.class_names = fire_cls
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
        cv2.moveWindow(win_name, 50, 480)  # 左下角
        
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