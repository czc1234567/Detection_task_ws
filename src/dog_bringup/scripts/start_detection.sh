#!/usr/bin/env bash
# =====================================================================
# 巡检检测一键启动 (标准消息, 无自定义消息)
#
# 任务选择 (可选一个):
#   bash start_detection.sh                  # 启动全部任务 (停车 + PPE)
#   bash start_detection.sh parking          # 仅车辆违规停放 + 行人
#   bash start_detection.sh ppe              # 仅安全帽 / 安全服 (工装)
#   bash start_detection.sh all              # 显式指定启动全部任务
#
# 常用调试与显示选项 (标志可放在任意位置, 支持组合使用):
#   bash start_detection.sh --gui            # 打开 OpenCV 本地画面窗口
#   bash start_detection.sh --no-rviz        # 不打开 RViz2 界面 (终端无头运行)
#   bash start_detection.sh parking --gui    # 仅车辆任务 + 打开 OpenCV 窗口
#   bash start_detection.sh ppe --no-rviz    # 仅 PPE 任务 + 不打开 RViz2
#
# 性能优化 / 纯业务运行选项 (极致省 CPU/GPU 算力):
#   bash start_detection.sh --no-markers     # 关闭 3D Marker 发布 (不渲染 3D 空间框)
#   bash start_detection.sh --no-image       # 关闭画面压缩流发布 (仅告警与数据上报)
#   bash start_detection.sh --no-rviz --no-markers  # 纯后台推理, 最流畅模式
# =====================================================================

set -e

TASK=all
NO_RVIZ=0
SHOW_GUI=0
PUB_RVZ_MARKERS=1
PUB_IMAGE=1

# 解析所有传入参数
for a in "$@"; do
  case "$a" in
    --no-rviz) NO_RVIZ=1 ;;
    --gui) SHOW_GUI=1 ;;
    --no-markers) PUB_RVZ_MARKERS=0 ;;
    --no-image) PUB_IMAGE=0 ;;
    parking|ppe|all) TASK=$a ;;
  esac
done

ROS_ARGS=""
[ $SHOW_GUI -eq 1 ] && ROS_ARGS="$ROS_ARGS -p show_gui:=true"
[ $PUB_RVZ_MARKERS -eq 0 ] && ROS_ARGS="$ROS_ARGS -p publish_rviz_markers:=false"
[ $PUB_IMAGE -eq 0 ] && ROS_ARGS="$ROS_ARGS -p publish_image:=false"

CONDA_PY=/home/jetson/anaconda3/envs/yolo26/bin/python3
WS=/home/jetson/Detection_task_ws
PARAMS=$WS/install/dog_bringup/share/dog_bringup/config/params.yaml
RVZ_CFG=$WS/install/dog_bringup/share/dog_bringup/config/rviz_detection.rviz

source /opt/ros/humble/setup.bash
source $WS/install/setup.bash

echo "=== 检测启动 (task=$TASK, gui=$SHOW_GUI, rviz=$([ $NO_RVIZ -eq 0 ] && echo on || echo off), markers=$PUB_RVZ_MARKERS, image=$PUB_IMAGE) ==="

# 静态 TF: map -> camera_link (驱动已发布 camera_link -> camera_color_optical_frame)
ros2 run tf2_ros static_transform_publisher \
  0 0 0 0 0 0 map camera_link > /dev/null 2>&1 &
PIDS=$!

# 检测节点 1: 车辆违规停放 + 行人 (conda yolo26 环境)
if [ "$TASK" = "all" ] || [ "$TASK" = "parking" ]; then
  $CONDA_PY -m dog_ai_detection.parking_detection_node --ros-args \
    -r __node:=parking_detection_node --params-file $PARAMS $ROS_ARGS &
  PIDS="$PIDS $!"
fi

# 检测节点 2: 安全帽 / 安全工装 (conda yolo26 环境)
if [ "$TASK" = "all" ] || [ "$TASK" = "ppe" ]; then
  $CONDA_PY -m dog_ai_detection.helmet_vest_detection_node --ros-args \
    -r __node:=helmet_vest_detection_node --params-file $PARAMS $ROS_ARGS &
  PIDS="$PIDS $!"
fi

# RViz2 可视化界面
if [ $NO_RVIZ -eq 0 ]; then
  ros2 run rviz2 rviz2 -d $RVZ_CFG &
  PIDS="$PIDS $!"
fi

echo "=== 已启动, Ctrl+C 停止 ==="
trap 'kill $PIDS 2>/dev/null' INT TERM
wait