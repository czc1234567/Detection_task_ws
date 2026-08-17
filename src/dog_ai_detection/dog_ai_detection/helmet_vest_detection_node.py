#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dog_ai_detection.helmet_vest_detection_node
===========================================
节点 2: 安全帽 / 安全服 (工装) 检测.

模型: models/helmet.pt (类别: Helm / No_Helm / No_Vest / Vest)
    - Helm / Vest       正常 (橙框)
    - No_Helm / No_Vest 违规 (红框 + /detection/alert 告警)

注: 模型直接输出违规类别, 无需额外规则. 如需扩展 (如人员未佩戴判定),
    覆盖 _check_violation() 即可.

话题 (标准消息, 无自定义消息):
    订阅: /camera/color/image_raw, /camera/depth/image_raw, /camera/color/camera_info
    发布: /detection/ppe_result_image, /detection/object_markers,
          /detection/alert, /detection/web_json

启动: 见 dog_bringup/scripts/start_detection.sh
"""

import rclpy

from dog_ai_detection.detection_core import BaseVizDetectionNode


class HelmetVestDetectionNode(BaseVizDetectionNode):
    """安全帽 / 安全服检测节点."""

    TASK_NODE_NAME = 'helmet_vest_detection_node'
    # helmet.engine 与当前 ultralytics 版本不兼容, 优先使用 helmet.pt
    MODEL_CANDIDATES = ('helmet.pt', 'helmet.engine')

    def _declare_task_params(self):
        self.declare_parameter('result_image_topic', '/detection/ppe_result_image')
        self.declare_parameter('ppe_classes', ['Helm', 'No_Helm', 'No_Vest', 'Vest'])
        # conf_threshold 等公共参数由基类声明, 通过 params.yaml 配置 (0.4)

    def build_groups(self):
        self.ppe_classes = list(self.get_parameter('ppe_classes').value)
        return [
            {'name': 'ppe', 'classes': self.ppe_classes,
             'offset': 2000, 'zones': False,
             # 类别即违规: 模型输出的 No_Helm / No_Vest
             'violation_classes': ('No_Helm', 'No_Vest'),
             'color_ok': (1.0, 0.6, 0.0), 'color_vio': (1.0, 0.0, 0.0),
             'color_ok_bgr': (0, 165, 255), 'color_vio_bgr': (0, 0, 255)},
        ]


def main(args=None):
    rclpy.init(args=args)
    node = HelmetVestDetectionNode()
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
