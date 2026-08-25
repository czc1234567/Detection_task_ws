#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dog_ai_detection.yolo_tracker
=============================
2D 目标多状态追踪与违规消抖模块
支持:
1. ByteTrack (低分框二次关联，适合机器狗抖动失焦)
2. SmoothKalmanTrack (带运动平滑预测的轻量卡尔曼)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(boxes1, boxes2):
    """计算两个 BBox 列表的 IoU 矩阵: boxes 格式均为 [[x1, y1, x2, y2], ...]"""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = np.array(boxes1, dtype=np.float32)
    b2 = np.array(boxes2, dtype=np.float32)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])

    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter

    return inter / np.clip(union, 1e-6, None)


# -------------------------------------------------------------
# 2D 卡尔曼滤波 (针对机身抖动做了测量噪声平滑)
# -------------------------------------------------------------
class KalmanBoxTracker2D:
    """平滑 2D Bbox: 状态量为 [cx, cy, s(面积), r(长宽比), v_cx, v_cy, v_s]"""
    def __init__(self, bbox):
        self.x = np.zeros((7, 1))
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        self.x[0] = bbox[0] + w / 2.0
        self.x[1] = bbox[1] + h / 2.0
        self.x[2] = w * h
        self.x[3] = w / h

        # 针对机器狗颠簸，增大过程噪声 Q 允许更快适应突变，合理设置观测噪声 R
        self.P = np.diag([10., 10., 10., 10., 1e3, 1e3, 1e3])
        self.Q = np.diag([2., 2., 5., 1e-2, 1e-1, 1e-1, 1e-1])
        self.R = np.diag([2., 2., 10., 10.])
        
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_state()

    def update(self, bbox):
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        z = np.array([[bbox[0] + w / 2.0], [bbox[1] + h / 2.0], [w * h], [w / h]])
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self):
        w = np.sqrt(max(1.0, self.x[2, 0] * self.x[3, 0]))
        h = self.x[2, 0] / (w + 1e-6)
        return [
            float(self.x[0, 0] - w / 2.0),
            float(self.x[1, 0] - h / 2.0),
            float(self.x[0, 0] + w / 2.0),
            float(self.x[1, 0] + h / 2.0)
        ]


# -------------------------------------------------------------
# 跟踪目标元数据结构 (包含连续违规消抖计数)
# -------------------------------------------------------------
class DetectionTrack:
    def __init__(self, track_id, det_dict, is_violation=False, violation_type=None):
        self.track_id = track_id
        self.kf = KalmanBoxTracker2D(det_dict['bbox'])
        self.class_name = det_dict['class_name']
        self.score = float(det_dict['score'])
        self.latest_det = det_dict
        
        self.misses = 0          # 丢失帧数计数
        self.total_frames = 1    # 跟踪存活帧数
        
        # 违规连续帧消抖计数器
        self.violation_consecutive_hits = 1 if is_violation else 0
        self.is_confirmed_violation = False
        self.current_violation_type = violation_type

    def update(self, det_dict, is_violation=False, violation_type=None):
        self.kf.update(det_dict['bbox'])
        self.class_name = det_dict['class_name']
        self.score = float(det_dict['score'])
        self.latest_det = det_dict
        self.misses = 0
        self.total_frames += 1
        
        if is_violation:
            self.violation_consecutive_hits += 1
            self.current_violation_type = violation_type
        else:
            # 中断时衰减或重置
            self.violation_consecutive_hits = max(0, self.violation_consecutive_hits - 1)
            if self.violation_consecutive_hits == 0:
                self.current_violation_type = None
                self.is_confirmed_violation = False

    def mark_miss(self):
        self.misses += 1


# -------------------------------------------------------------
# 算法 1: ByteTrack (原生 2D 两阶段关联 + 违规判定过滤)
# -------------------------------------------------------------
class ByteTracker2D:
    def __init__(self, min_hits=15, max_miss=5, high_thresh=0.4, low_thresh=0.15, iou_thresh=0.3):
        self.min_hits = min_hits
        self.max_miss = max_miss
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.tracks = {}
        self.next_id = 0

    def step(self, raw_dets, violation_eval_fn):
        """
        raw_dets: YOLO 原生输出 [{'class_name': str, 'score': float, 'bbox': [x1, y1, x2, y2]}, ...]
        violation_eval_fn: 传入 det 字典，返回 (is_violation: bool, violation_name: str/None)
        """
        # 1. 预测所有已有轨迹
        for trk in self.tracks.values():
            trk.kf.predict()

        # 2. 按置信度分流（应对机器狗颠簸画面模糊）
        high_dets = []
        low_dets = []
        for d in raw_dets:
            if d['score'] >= self.high_thresh:
                high_dets.append(d)
            elif d['score'] >= self.low_thresh:
                low_dets.append(d)

        matched_track_ids = set()
        matched_high_idx = set()

        # 3. 第一轮关联：高分检测框与预测 Track 匹配
        if self.tracks and high_dets:
            track_ids = list(self.tracks.keys())
            pred_boxes = [self.tracks[tid].kf.get_state() for tid in track_ids]
            det_boxes = [d['bbox'] for d in high_dets]
            
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes)
            # 增加类别过滤（同类别才匹配）
            for i, tid in enumerate(track_ids):
                for j, d in enumerate(high_dets):
                    if self.tracks[tid].class_name != d['class_name']:
                        iou_mat[i, j] = 0.0

            r_idx, c_idx = linear_sum_assignment(-iou_mat)
            for r, c in zip(r_idx, c_idx):
                if iou_mat[r, c] >= self.iou_thresh:
                    tid = track_ids[r]
                    det = high_dets[c]
                    is_vio, v_name = violation_eval_fn(det)
                    self.tracks[tid].update(det, is_vio, v_name)
                    matched_track_ids.add(tid)
                    matched_high_idx.add(c)

        # 4. 第二轮关联：剩余 Track 与低分框匹配 (挽救颠簸目标)
        unmatched_track_ids = [tid for tid in self.tracks if tid not in matched_track_ids]
        if unmatched_track_ids and low_dets:
            pred_boxes = [self.tracks[tid].kf.get_state() for tid in unmatched_track_ids]
            det_boxes = [d['bbox'] for d in low_dets]
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes)
            for i, tid in enumerate(unmatched_track_ids):
                for j, d in enumerate(low_dets):
                    if self.tracks[tid].class_name != d['class_name']:
                        iou_mat[i, j] = 0.0

            r_idx, c_idx = linear_sum_assignment(-iou_mat)
            for r, c in zip(r_idx, c_idx):
                if iou_mat[r, c] >= self.iou_thresh:
                    tid = unmatched_track_ids[r]
                    det = low_dets[c]
                    is_vio, v_name = violation_eval_fn(det)
                    self.tracks[tid].update(det, is_vio, v_name)
                    matched_track_ids.add(tid)

        # 5. 未匹配的高分检测新建 Track
        for j, det in enumerate(high_dets):
            if j not in matched_high_idx:
                is_vio, v_name = violation_eval_fn(det)
                self.tracks[self.next_id] = DetectionTrack(self.next_id, det, is_vio, v_name)
                matched_track_ids.add(self.next_id)
                self.next_id += 1

        # 6. 清理失联轨迹 & 提取连续达到 min_hits 帧的有效违规
        confirmed_violations = []
        all_active_tracks = []
        del_ids = []

        for tid, trk in self.tracks.items():
            if tid not in matched_track_ids:
                trk.mark_miss()
                if trk.misses > self.max_miss:
                    del_ids.append(tid)
            else:
                all_active_tracks.append(trk)
                # 核心逻辑：连续达到指定帧数（如15帧）违规判定才成立
                if trk.violation_consecutive_hits >= self.min_hits:
                    trk.is_confirmed_violation = True
                    confirmed_violations.append({
                        'track_id': trk.track_id,
                        'class_name': trk.class_name,
                        'violation': trk.current_violation_type,
                        'confidence': round(trk.score, 3),
                        'bbox_2d': [int(v) for v in trk.latest_det['bbox'][:4]],
                        'consecutive_hits': trk.violation_consecutive_hits
                    })

        for tid in del_ids:
            del self.tracks[tid]

        return all_active_tracks, confirmed_violations


# -------------------------------------------------------------
# 算法 2: SmoothKalmanTrack (标准轻量级单阶段卡尔曼跟踪)
# -------------------------------------------------------------
class SmoothKalmanTracker2D:
    def __init__(self, min_hits=15, max_miss=5, iou_thresh=0.3):
        self.min_hits = min_hits
        self.max_miss = max_miss
        self.iou_thresh = iou_thresh
        self.tracks = {}
        self.next_id = 0

    def step(self, raw_dets, violation_eval_fn):
        for trk in self.tracks.values():
            trk.kf.predict()

        matched_track_ids = set()
        matched_det_idx = set()

        if self.tracks and raw_dets:
            track_ids = list(self.tracks.keys())
            pred_boxes = [self.tracks[tid].kf.get_state() for tid in track_ids]
            det_boxes = [d['bbox'] for d in raw_dets]
            
            iou_mat = compute_iou_matrix(pred_boxes, det_boxes)
            for i, tid in enumerate(track_ids):
                for j, d in enumerate(raw_dets):
                    if self.tracks[tid].class_name != d['class_name']:
                        iou_mat[i, j] = 0.0

            r_idx, c_idx = linear_sum_assignment(-iou_mat)
            for r, c in zip(r_idx, c_idx):
                if iou_mat[r, c] >= self.iou_thresh:
                    tid = track_ids[r]
                    det = raw_dets[c]
                    is_vio, v_name = violation_eval_fn(det)
                    self.tracks[tid].update(det, is_vio, v_name)
                    matched_track_ids.add(tid)
                    matched_det_idx.add(c)

        for j, det in enumerate(raw_dets):
            if j not in matched_det_idx:
                is_vio, v_name = violation_eval_fn(det)
                self.tracks[self.next_id] = DetectionTrack(self.next_id, det, is_vio, v_name)
                matched_track_ids.add(self.next_id)
                self.next_id += 1

        confirmed_violations = []
        all_active_tracks = []
        del_ids = []

        for tid, trk in self.tracks.items():
            if tid not in matched_track_ids:
                trk.mark_miss()
                if trk.misses > self.max_miss:
                    del_ids.append(tid)
            else:
                all_active_tracks.append(trk)
                if trk.violation_consecutive_hits >= self.min_hits:
                    trk.is_confirmed_violation = True
                    confirmed_violations.append({
                        'track_id': trk.track_id,
                        'class_name': trk.class_name,
                        'violation': trk.current_violation_type,
                        'confidence': round(trk.score, 3),
                        'bbox_2d': [int(v) for v in trk.latest_det['bbox'][:4]],
                        'consecutive_hits': trk.violation_consecutive_hits
                    })

        for tid in del_ids:
            del self.tracks[tid]

        return all_active_tracks, confirmed_violations


def build_tracker(tracker_type: str, min_hits: int = 15, max_miss: int = 5):
    """工厂接口：根据配置实例化对应算法"""
    name = str(tracker_type).lower().strip()
    if name == 'bytetrack':
        return ByteTracker2D(min_hits=min_hits, max_miss=max_miss)
    else:
        return SmoothKalmanTracker2D(min_hits=min_hits, max_miss=max_miss)