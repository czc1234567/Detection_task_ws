#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点 1: 车辆违规停放与行人检测."""

import sys
import time
import signal
import threading
import numpy as np
import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from dog_ai_detection.detection_core import BaseVizDetectionNode


class ParkingDetectionNode(BaseVizDetectionNode):
    TASK_NODE_NAME = 'parking_detection_node'
    MODEL_CANDIDATES = ('yolo26n.engine', 'yolo26n.pt')

    def _declare_task_params(self):
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
             'offset': 1000, 'zones': True,
             'color_ok': (0.3, 0.6, 1.0), 'color_vio': (1.0, 0.0, 0.0),
             'color_ok_bgr': (255, 140, 30), 'color_vio_bgr': (0, 0, 255)},
        ]


def main(args=None):
    rclpy.init(args=args)
    node = ParkingDetectionNode()

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
        cv2.moveWindow(win_name, 50, 50)  # 左上角
        
        # 刷入初始帧，立刻激活 X11 窗口渲染
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
                # 无论是否拿到新帧，必须调用 waitKey 驱动 X11 事件循环
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