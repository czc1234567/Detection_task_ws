#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dog_ai_detection.detection_core
===============================
共享检测核心 (仅标准消息, 无自定义消息):

    YoloDetector          多后端推理引擎 (ultralytics / TensorRT / ONNX)
    BaseVizDetectionNode  可视化检测节点基类: 彩色+深度同步订阅 -> 推理 ->
                          像素+深度->3D -> TF->map -> 任务组标记/MarkerArray
                          -> 画面标注 -> OpenCV 窗口 / RViz 画面

任务拆分架构 (后续叠加新检测任务只需新增一个子类文件 + 脚本入口):
    yolo26n (COCO)  -> parking_detection_node     车辆违规停放 + 行人(测试)
    helmet.pt (PPE) -> helmet_vest_detection_node 安全帽 / 安全服
    future_model    -> xxx_detection_node          后续任务...
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# 包内模型目录 (模型文件随 Python 包一起安装)
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


class YoloDetector:
    """多后端 YOLO 推理引擎.

    infer() 统一返回:
        [{'class_name': str, 'score': float, 'bbox': [x1, y1, x2, y2]}, ...]
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, class_names=None,
                 device: str = '', imgsz: int = 640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = list(class_names or [])
        self.device = device
        self.imgsz = imgsz

        self._backend = None
        self._names = {i: n for i, n in enumerate(self.class_names)}
        self.last_error = ''

        self._model = None
        self._ort = None
        self._engine = None
        self._context = None
        self._cuda = None
        self._cuda_ctx = None
        self._trt = None
        self._input_shape = None
        self._trt_bindings = None
        self._trt_input_dev = None
        self._trt_input_dtype = None
        self._trt_output_dev = None
        self._trt_output_shape = None
        self._trt_output_dtype = None
        self._ort_input_name = None

    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == '.engine':
            ok = self._try_ultralytics() or self._try_tensorrt()
        elif ext == '.onnx':
            ok = self._try_onnxruntime()
        else:
            ok = self._try_ultralytics()
        return ok

    # ============================ ultralytics ============================
    def _try_ultralytics(self) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError:
            self.last_error = ('ultralytics 后端需要 PyTorch, 请安装: '
                               'pip install ultralytics (Jetson 需 NVIDIA 提供的 torch wheel)')
            return False
        try:
            self._model = YOLO(self.model_path, task='detect')
            names = getattr(self._model, 'names', None)
            if isinstance(names, dict):
                self._names = {int(k): str(v) for k, v in names.items()}
            self._backend = 'ultralytics'
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f'ultralytics 加载模型失败: {exc}'
            return False

    def _infer_ultralytics(self, bgr):
        results = self._model.predict(
            source=bgr, conf=self.conf_threshold, iou=self.iou_threshold,
            imgsz=self.imgsz, device=self.device or None, verbose=False)
        dets = []
        if not results:
            return dets
        boxes = results[0].boxes
        if boxes is None:
            return dets
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            dets.append({
                'class_name': self._names.get(cls_id, str(cls_id)),
                'score': float(boxes.conf[i].item()),
                'bbox': [float(v) for v in boxes.xyxy[i].tolist()],
            })
        return dets

    # ============================= TensorRT =============================
    def _try_tensorrt(self) -> bool:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as exc:
            missing = getattr(exc, 'name', '')
            self.last_error = (f'原生 TensorRT 后端缺少依赖 {missing or exc}, '
                               '请执行: pip install pycuda')
            return False
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, 'rb') as f:
                engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError('deserialize_cuda_engine 返回 None')
            self._cuda = cuda
            self._trt = trt
            # 显式创建 CUDA Context (不使用 autoinit, 保证多线程执行器下安全)
            self._cuda_ctx = cuda.Device(0).make_context()
            self._engine = engine
            self._context = engine.create_execution_context()
            self._trt_bindings = []
            self._setup_trt_bindings()
            self._backend = 'tensorrt'
            self._cuda_ctx.pop()   # 归还上下文栈, infer() 时再 push
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f'TensorRT 加载失败: {exc}'
            return False

    def _setup_trt_bindings(self):
        trt = self._trt
        dev_ptrs = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = list(self._engine.get_tensor_shape(name))
            mode = self._engine.get_tensor_mode(name)
            nptype = trt.nptype(self._engine.get_tensor_dtype(name))
            shape = [1 if s < 0 else s for s in shape]
            size = int(np.prod(shape)) * np.dtype(nptype).itemsize
            dev = self._cuda.mem_alloc(size)
            if mode == trt.TensorIOMode.INPUT:
                self._input_shape = tuple(shape)
                self._trt_input_dtype = nptype
                self._trt_input_dev = dev
            else:
                self._trt_output_shape = tuple(shape)
                self._trt_output_dtype = nptype
                self._trt_output_dev = dev
            dev_ptrs.append(int(dev))
        self._trt_bindings = dev_ptrs

    def _infer_tensorrt(self, bgr):
        ih, iw = self._input_shape[2], self._input_shape[3]
        img, ratio, (dw, dh) = self._letterbox(bgr, (iw, ih))
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None]).astype(self._trt_input_dtype)
        # 显式绑定 CUDA Context, 保证任意线程调用 infer 都安全
        self._cuda_ctx.push()
        try:
            self._cuda.memcpy_htod(self._trt_input_dev, blob)
            self._context.execute_v2(self._trt_bindings)
            out = np.empty(self._trt_output_shape, dtype=self._trt_output_dtype)
            self._cuda.memcpy_dtoh(out, self._trt_output_dev)
        finally:
            self._cuda_ctx.pop()
        return self._decode_yolo_output(out, ratio, dw, dh, bgr.shape)

    # =========================== onnxruntime ============================
    def _try_onnxruntime(self) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            self.last_error = 'onnxruntime 未安装: pip install onnxruntime-gpu'
            return False
        try:
            self._ort = ort.InferenceSession(
                self.model_path,
                providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider',
                           'CPUExecutionProvider'])
            in_meta = self._ort.get_inputs()[0]
            self._ort_input_name = in_meta.name
            shape = list(in_meta.shape)
            ih = int(shape[2]) if len(shape) > 2 and isinstance(shape[2], int) else self.imgsz
            iw = int(shape[3]) if len(shape) > 3 and isinstance(shape[3], int) else self.imgsz
            self._input_shape = (1, 3, ih, iw)
            self._backend = 'onnxruntime'
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f'onnxruntime 加载模型失败: {exc}'
            return False

    def _infer_onnxruntime(self, bgr):
        ih, iw = self._input_shape[2], self._input_shape[3]
        img, ratio, (dw, dh) = self._letterbox(bgr, (iw, ih))
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None])
        out = self._ort.run(None, {self._ort_input_name: blob})[0]
        return self._decode_yolo_output(out, ratio, dw, dh, bgr.shape)

    # ------------------------------ 公共解码 ------------------------------
    @staticmethod
    def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (new_shape[0] - nw) // 2, (new_shape[1] - nh) // 2
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_shape[1], new_shape[0], 3), color, dtype=np.uint8)
        canvas[dh:dh + nh, dw:dw + nw] = resized
        return canvas, r, (dw, dh)

    def _decode_yolo_output(self, out, ratio, dw, dh, orig_shape):
        dets = []
        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2:
            return dets
        if out.shape[0] < out.shape[1]:
            out = out.T
        boxes = out[:, :4]
        scores = out[:, 4:]
        if scores.size == 0:
            return dets
        conf = scores.max(axis=1)
        cls = scores.argmax(axis=1)
        keep = conf >= self.conf_threshold
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        if len(conf) == 0:
            return dets

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / ratio
        h, w = orig_shape[:2]
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)

        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), conf.tolist(),
                                   self.conf_threshold, self.iou_threshold)
        for idx in (indices.flatten() if len(indices) else []):
            cls_id = int(cls[idx])
            dets.append({
                'class_name': self._names.get(cls_id, str(cls_id)),
                'score': float(conf[idx]),
                'bbox': [float(v) for v in xyxy[idx]],
            })
        return dets

    # ------------------------------ 统一入口 ------------------------------
    def infer(self, bgr) -> list:
        if self._backend == 'ultralytics':
            return self._infer_ultralytics(bgr)
        if self._backend == 'tensorrt':
            return self._infer_tensorrt(bgr)
        if self._backend == 'onnxruntime':
            return self._infer_onnxruntime(bgr)
        return []


class BaseVizDetectionNode(Node):
    """可视化检测节点基类 (标准消息: MarkerArray / Image / Bool / String).

    子类只需覆盖:
        TASK_NODE_NAME   节点名
        MODEL_CANDIDATES 模型文件候选 (包内 models/ 目录, 按顺序找)
        _declare_task_params()  声明本任务参数 (如类别表 / 画面话题)
        build_groups()          返回任务组列表:
            {'name': ..., 'classes': [...], 'offset': 标记ID偏移,
             'zones': 是否做禁停区判定,
             'violation_classes': 类别即违规 (如 No_Helm/No_Vest),
             'color_ok'/'color_vio' (RGB), 'color_ok_bgr'/'color_vio_bgr'}
        _check_violation()      可选, 覆盖自定义违规规则
    """

    TASK_NODE_NAME = 'detection_node'
    MODEL_CANDIDATES = ('yolo26n.engine', 'yolo26n.pt')

    def __init__(self):
        super().__init__(self.TASK_NODE_NAME)

        # ------------------------------ 参数 ------------------------------
        self._declare_task_params()
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('object_markers_topic', '/detection/object_markers')
        self.declare_parameter('zone_markers_topic', '/detection/map_markers')
        self.declare_parameter('alert_topic', '/detection/alert')
        self.declare_parameter('web_json_topic', '/detection/web_json')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('zone_frame', 'map')
        self.declare_parameter('model_path', '')
        self.declare_parameter('class_names', [])   # TRT/ONNX 后端 id->名称
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('marker_lifetime', 0.5)
        self.declare_parameter('embed_image_in_json', False)
        self.declare_parameter('show_gui', False)
        self.declare_parameter('gui_scale', 0.6)
        self.declare_parameter('result_image_scale', 0.6)
        self.declare_parameter('publish_image', True)
        self.declare_parameter('inference_period_ms', 100)
        self.declare_parameter('sync_slop', 0.2)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('max_depth', 8.0)
        self.declare_parameter('parking_zone', [])
        self.declare_parameter('fire_lane_zone', [])

        self._load_params()
        self.model_path = self._resolve_model(self.get_parameter('model_path').value)
        self._groups = self.build_groups()
        if not self._groups:
            self.get_logger().warn('任务组为空, 不会输出任何检测结果')

        # ------------------------------ 状态 ------------------------------
        self._bridge = CvBridge()
        self._color = None
        self._depth = None
        self._depth_is_meters = False
        self._header = None
        self._frame_dets = []
        self._fx = self._fy = self._cx = self._cy = None
        # 图像缓冲保护 (多线程执行器下防竞争; 单线程下也有序)
        self._frame_lock = threading.Lock()
        self._frame_seq = 0          # 新帧计数
        self._processed_seq = -1     # 已处理帧计数 (防重复处理/跳帧)
        # 异步 JPEG 编码 + JSON 上报 (避免阻塞检测主循环)
        self._encoder = ThreadPoolExecutor(max_workers=1)
        self._encode_busy = False

        from tf2_ros import Buffer, TransformListener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._engine = YoloDetector(
            model_path=self.model_path,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            class_names=self.class_names)
        if self._engine.load():
            self.get_logger().info(
                f'Model loaded (backend={self._engine._backend}): {self.model_path}')
        else:
            self.get_logger().error(
                f'Inference engine unavailable: {self._engine.last_error}')

        # ------------------------------ 通信 ------------------------------
        import message_filters
        from sensor_msgs.msg import CameraInfo
        self._cam_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb, 10)
        self._color_sub = message_filters.Subscriber(self, Image, self.color_topic)
        self._depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub], queue_size=2, slop=float(self.sync_slop))
        self._sync.registerCallback(self._image_sync_cb)

        from std_msgs.msg import Bool, String
        from visualization_msgs.msg import MarkerArray
        self._img_pub = self.create_publisher(Image, self.result_image_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, self.object_markers_topic, 10)
        self._zone_pub = self.create_publisher(MarkerArray, self.zone_markers_topic, 10)
        self._alert_pub = self.create_publisher(Bool, self.alert_topic, 10)
        self._json_pub = self.create_publisher(String, self.web_json_topic, 10)

        # ------------------------------ 定时器 ------------------------------
        self._timer = self.create_timer(self.inference_period_ms / 1000.0, self._process_tick)
        self._zone_timer = self.create_timer(1.0, self._publish_zones)

        self.get_logger().info(
            f'[{self.get_name()}] started: color={self.color_topic}, '
            f'depth={self.depth_topic}, groups='
            + ','.join(g['name'] for g in self._groups))

    # ------------------------------------------------------------------ #
    def _declare_task_params(self):
        """子类钩子: 声明任务参数 (在公共参数之前调用)."""

    def build_groups(self) -> list:
        """子类覆盖: 返回任务组列表."""
        return []

    def _check_violation(self, gname, det, mx, my):
        """违规判定: 类别即违规 > 禁停区域判定. 子类可覆盖."""
        g = self._group_by_name(gname)
        if g is None:
            return None
        if g.get('violation_classes') and det['class_name'] in g['violation_classes']:
            return det['class_name']
        if g.get('zones') and mx is not None:
            return self._check_zones(mx, my)
        return None

    def _group_by_name(self, gname):
        return next((g for g in self._groups if g['name'] == gname), None)

    # ------------------------------------------------------------------ #
    def _load_params(self):
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.object_markers_topic = self.get_parameter('object_markers_topic').value
        self.zone_markers_topic = self.get_parameter('zone_markers_topic').value
        self.alert_topic = self.get_parameter('alert_topic').value
        self.web_json_topic = self.get_parameter('web_json_topic').value
        self.result_image_topic = self.get_parameter('result_image_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.zone_frame = self.get_parameter('zone_frame').value
        self.class_names = list(self.get_parameter('class_names').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.marker_lifetime = float(self.get_parameter('marker_lifetime').value)
        self.embed_image_in_json = bool(self.get_parameter('embed_image_in_json').value)
        self.show_gui = bool(self.get_parameter('show_gui').value)
        self.gui_scale = float(self.get_parameter('gui_scale').value)
        self.result_image_scale = float(self.get_parameter('result_image_scale').value)
        self.publish_image = bool(self.get_parameter('publish_image').value)
        self.inference_period_ms = int(self.get_parameter('inference_period_ms').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.max_depth = float(self.get_parameter('max_depth').value)

        parking_flat = list(self.get_parameter('parking_zone').value)
        fire_flat = list(self.get_parameter('fire_lane_zone').value)
        self.parking_zone = [tuple(parking_flat[i:i + 2])
                             for i in range(0, len(parking_flat) - 1, 2)]
        self.fire_lane_zone = [tuple(fire_flat[i:i + 2])
                               for i in range(0, len(fire_flat) - 1, 2)]
        # 预转 numpy 数组 (cv2.pointPolygonTest 需要, C++ 底层判定)
        self.parking_zone_np = np.asarray(
            self.parking_zone, dtype=np.float32).reshape(-1, 1, 2)
        self.fire_lane_zone_np = np.asarray(
            self.fire_lane_zone, dtype=np.float32).reshape(-1, 1, 2)

    def _resolve_model(self, value: str) -> str:
        if value:
            return value if os.path.isabs(value) else os.path.join(MODELS_DIR, value)
        for name in self.MODEL_CANDIDATES:
            p = os.path.join(MODELS_DIR, name)
            if os.path.isfile(p):
                if name != self.MODEL_CANDIDATES[0]:
                    self.get_logger().warn(
                        f'未找到首选模型 {self.MODEL_CANDIDATES[0]}, 回退使用 {name} '
                        '(若检测类别与预期不符, 请放入训练好的专用模型)')
                return p
        return os.path.join(MODELS_DIR, self.MODEL_CANDIDATES[0])

    # ------------------------------------------------------------------ #
    # 回调: 相机内参 / 彩色+深度同步帧
    # ------------------------------------------------------------------ #
    def _camera_info_cb(self, msg):
        self._fx, self._fy = float(msg.k[0]), float(msg.k[4])
        self._cx, self._cy = float(msg.k[2]), float(msg.k[5])
        self.get_logger().info(
            f'Intrinsics: fx={self._fx:.2f} fy={self._fy:.2f} '
            f'cx={self._cx:.2f} cy={self._cy:.2f}', once=True)

    def _image_sync_cb(self, color_msg, depth_msg):
        try:
            color = self._bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth = self._bridge.imgmsg_to_cv2(depth_msg, '16UC1')
                is_meters = False
            elif depth_msg.encoding == '32FC1':
                depth = self._bridge.imgmsg_to_cv2(depth_msg, '32FC1')
                is_meters = True
            else:
                return
            with self._frame_lock:
                self._color = color
                self._depth = depth
                self._depth_is_meters = is_meters
                self._header = color_msg.header
                self._frame_seq += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Image sync conversion failed: {exc}')

    # ------------------------------------------------------------------ #
    # 几何工具: Marker / 多边形 / 深度 / 3D / TF
    # ------------------------------------------------------------------ #
    def _create_marker(self, mtype, mid, pose, scale, color, ns='default',
                       frame_id=None, text=''):
        from rclpy.duration import Duration
        from visualization_msgs.msg import Marker
        m = Marker()
        m.header.frame_id = frame_id or self.zone_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = mtype
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = float(pose[0]), float(pose[1])
        m.pose.position.z = float(pose[2]) if len(pose) > 2 else 0.0
        m.scale.x, m.scale.y, m.scale.z = float(scale[0]), float(scale[1]), float(scale[2])
        m.color.r, m.color.g, m.color.b, m.color.a = (
            float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        m.lifetime = (Duration(seconds=self.marker_lifetime).to_msg()
                      if self.marker_lifetime > 0 else Duration(seconds=0).to_msg())
        if text:
            m.text = text
        return m

    def _check_zones(self, x, y):
        """禁停区域判定 (cv2.pointPolygonTest, C++ 底层实现)."""
        if (len(self.parking_zone_np) >= 3 and
                cv2.pointPolygonTest(self.parking_zone_np, (float(x), float(y)), False) >= 0):
            return 'illegal_parking'
        if (len(self.fire_lane_zone_np) >= 3 and
                cv2.pointPolygonTest(self.fire_lane_zone_np, (float(x), float(y)), False) >= 0):
            return 'fire_lane_occupation'
        return None

    def _depth_at_center(self, x1, y1, x2, y2):
        with self._frame_lock:
            if self._depth is None:
                return None
            h, w = self._depth.shape[:2]
            cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
            cy = int(np.clip((y1 + y2) / 2, 0, h - 1))
            roi = self._depth[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3].copy()
        valid = roi[roi > 0]
        if len(valid) == 0:
            return None
        value = float(np.median(valid))
        return value if self._depth_is_meters else value * self.depth_scale

    def _to_camera_3d(self, x1, y1, x2, y2, depth_m):
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return ((u - self._cx) * depth_m / self._fx,
                (v - self._cy) * depth_m / self._fy,
                depth_m)

    def _lookup_frame_tf(self):
        """每帧只调用一次: 缓存 camera->map 变换, 失败返回 None."""
        import rclpy.time
        try:
            return self._tf_buffer.lookup_transform(
                self.map_frame, self.camera_frame, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'TF {self.camera_frame}->{self.map_frame} failed: {exc}',
                throttle_duration_sec=5.0)
            return None

    def _transform_point_to_map(self, px, py, pz, frame_tf):
        """用已缓存的变换把单个相机 3D 点转到 map (不触发 TF 查询)."""
        from geometry_msgs.msg import Point, PointStamped
        from tf2_geometry_msgs import do_transform_point
        pt = PointStamped()
        pt.header.frame_id = self.camera_frame
        pt.header.stamp = frame_tf.header.stamp
        pt.point = Point(x=float(px), y=float(py), z=float(pz))
        out = do_transform_point(pt, frame_tf)
        return out.point.x, out.point.y, out.point.z

    # ------------------------------------------------------------------ #
    # 定时: 一次推理 -> 分组提取 -> 画面标注 -> 3D 标记 -> 告警
    # ------------------------------------------------------------------ #
    def _process_tick(self):
        if self._color is None or self._fx is None:
            return
        with self._frame_lock:
            if self._frame_seq == self._processed_seq:
                return                    # 无新帧, 不重复处理
            frame = self._color.copy()
            self._processed_seq = self._frame_seq

        # 模型每帧只推理一次
        raw_dets = self._engine.infer(frame)
        self._frame_dets = raw_dets
        # 单帧只查询一次 TF (失败则本帧跳过 3D/Map 计算)
        frame_tf = self._lookup_frame_tf()
        groups = {g['name']: [d for d in raw_dets if d['class_name'] in g['classes']]
                  for g in self._groups}

        from std_msgs.msg import Bool
        from visualization_msgs.msg import Marker, MarkerArray
        import json
        from std_msgs.msg import String
        markers = MarkerArray()
        violations = []
        alert = False

        # 画面: 全部检测画灰色细框
        for det in raw_dets:
            x1, y1, x2, y2 = [int(v) for v in det['bbox'][:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(frame, det['class_name'], (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)

        for gname, dets in groups.items():
            g = self._group_by_name(gname)
            for i, det in enumerate(dets):
                x1, y1, x2, y2 = [int(v) for v in det['bbox'][:4]]

                depth_m = self._depth_at_center(x1, y1, x2, y2)
                if depth_m is None or depth_m > self.max_depth:
                    depth_m = None
                mapp = None
                if depth_m is not None and frame_tf is not None:
                    cam = self._to_camera_3d(x1, y1, x2, y2, depth_m)
                    mapp = self._transform_point_to_map(*cam, frame_tf)
                mx, my, mz = mapp if mapp is not None else (None, None, None)

                zone = self._check_violation(gname, det, mx, my)
                is_violation = zone is not None
                if is_violation:
                    alert = True

                # 画面标注 (分组彩色框)
                c = g['color_vio_bgr'] if is_violation else g['color_ok_bgr']
                label = "{} {:.2f}".format(det['class_name'], float(det['score']))
                if depth_m is not None:
                    label += " {:.2f}m".format(depth_m)
                if zone:
                    label += " [" + zone + "]"
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                ty = y2 + 18 if y2 + 18 < frame.shape[0] else y1 - 6
                cv2.putText(frame, label, (x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

                # 3D 标记 (需要有效深度 + TF)
                if mapp is not None:
                    bw = (x2 - x1) * depth_m / self._fx
                    bh = (y2 - y1) * depth_m / self._fy
                    rgba = (g['color_vio'] + (0.8,) if is_violation
                            else g['color_ok'] + (0.7,))
                    base = g['offset'] + i * 2
                    markers.markers.append(self._create_marker(
                        Marker.CUBE, base, (mx, my, depth_m / 2.0),
                        (max(bw, 0.2), max(bh, 0.2), 0.2), rgba,
                        ns=gname + '_objects', frame_id=self.map_frame))
                    markers.markers.append(self._create_marker(
                        Marker.TEXT_VIEW_FACING, base + 1,
                        (mx, my, depth_m / 2.0 + 0.35), (0, 0, 0.35),
                        (1.0, 1.0, 1.0, 1.0), ns=gname + '_labels',
                        frame_id=self.map_frame, text=label))

                if is_violation:
                    violations.append({
                        'class_name': det['class_name'],
                        'violation': zone,
                        'confidence': round(float(det['score']), 3),
                        'position': ({'x': round(mx, 3), 'y': round(my, 3),
                                      'z': round(mz, 3)} if mapp is not None else None),
                    })

        self._marker_pub.publish(markers)

        if self.show_gui:
            self._show_gui(frame)
        if self.publish_image:
            self._img_pub.publish(self._to_image_msg(
                self._resize(frame, self.result_image_scale)))

        if alert:
            self._alert_pub.publish(Bool(data=True))
            self.get_logger().warn(f'{len(violations)} violation(s) detected')

        if violations:
            payload = {
                'type': 'violations',
                'timestamp': int(self.get_clock().now().nanoseconds / 1e9),
                'violations': violations,
            }
            if self.embed_image_in_json:
                # 异步 JPEG 编码 + 发布 (后台线程, 忙时丢弃本帧防积压)
                if not self._encode_busy:
                    self._encode_busy = True
                    self._encoder.submit(
                        self._encode_and_publish, frame.copy(), payload)
            else:
                self._json_pub.publish(
                    String(data=json.dumps(payload, ensure_ascii=False)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize(frame, scale):
        if scale >= 1.0:
            return frame
        w = max(1, int(frame.shape[1] * scale))
        h = max(1, int(frame.shape[0] * scale))
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    def _show_gui(self, frame):
        view = self._resize(frame, self.gui_scale)
        cv2.imshow('Detection - {}'.format(self.get_name()), view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('GUI: q pressed, shutting down')
            raise KeyboardInterrupt

    def _encode_and_publish(self, frame, payload):
        """后台线程: JPEG 压缩 + Base64 + JSON 发布 (不阻塞检测主循环)."""
        from std_msgs.msg import String
        import json
        try:
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                import base64
                payload['image_base64'] = base64.b64encode(buf).decode('utf-8')
            self._json_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Async JSON publish failed: {exc}')
        finally:
            self._encode_busy = False

    def destroy_node(self):
        if getattr(self, '_encoder', None) is not None:
            self._encoder.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def _to_image_msg(self, frame):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.strides[0]
        msg.data = frame.tobytes()
        return msg

    # ------------------------------------------------------------------ #
    # 定时: 发布禁停区域轮廓 (仅停车任务配置区域时可见)
    # ------------------------------------------------------------------ #
    def _publish_zones(self):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray
        arr = MarkerArray()
        zones = [('parking', self.parking_zone, (0.0, 1.0, 0.0, 1.0)),
                 ('fire_lane', self.fire_lane_zone, (1.0, 0.6, 0.0, 1.0))]
        for name, poly, color in zones:
            if len(poly) < 3:
                continue
            mid = len(arr.markers)
            line = self._create_marker(
                Marker.LINE_STRIP, mid, (0, 0, 0), (0.15, 0, 0), color,
                ns=name, frame_id=self.zone_frame)
            for p in poly:
                line.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.05))
            line.points.append(line.points[0])
            arr.markers.append(line)
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            arr.markers.append(self._create_marker(
                Marker.TEXT_VIEW_FACING, mid + 100, (cx, cy, 0.6), (0, 0, 0.8),
                (1.0, 1.0, 1.0, 1.0), ns=name + '_text',
                frame_id=self.zone_frame, text=name))
        self._zone_pub.publish(arr)
