#!/usr/bin/env bash
# =====================================================================
# 巡检检测一键启动 (标准消息, 无自定义消息)
#
# 用法:
#   bash start_detection.sh                  # 停车 + PPE 全部
#   bash start_detection.sh parking          # 仅车辆违规停放 + 行人
#   bash start_detection.sh ppe              # 仅安全帽/安全服
#   bash start_detection.sh parking --gui    # 同时打开 OpenCV 窗口
#   bash start_detection.sh --no-rviz        # 不打开 RViz
#   (标志可放在任意位置)
# =====================================================================

set -e

TASK=all
NO_RVIZ=0
SHOW_GUI=0
for a in "$@"; do
  case "$a" in
    --no-rviz) NO_RVIZ=1 ;;
    --gui) SHOW_GUI=1 ;;
    parking|ppe|all) TASK=$a ;;
  esac
done

GUI_ARGS=""
[ $SHOW_GUI -eq 1 ] && GUI_ARGS="-p show_gui:=true"

CONDA_PY=/home/jetson/anaconda3/envs/yolo26/bin/python3
WS=/home/jetson/Detection_task_ws
PARAMS=$WS/install/dog_bringup/share/dog_bringup/config/params.yaml
RVZ_CFG=$WS/install/dog_bringup/share/dog_bringup/config/rviz_detection.rviz

source /opt/ros/humble/setup.bash
source $WS/install/setup.bash

echo "=== 检测启动 (task=$TASK, gui=$SHOW_GUI, rviz=$([ $NO_RVIZ -eq 0 ] && echo on || echo off)) ==="

# 静态 TF: map -> camera_link (驱动已发布 camera_link -> camera_color_optical_frame)
ros2 run tf2_ros static_transform_publisher \
  0 0 0 0 0 0 map camera_link > /dev/null 2>&1 &
PIDS=$!

# 检测节点 (conda yolo26 环境)
if [ "$TASK" = "all" ] || [ "$TASK" = "parking" ]; then
  $CONDA_PY -m dog_ai_detection.parking_detection_node --ros-args \
    -r __node:=parking_detection_node --params-file $PARAMS $GUI_ARGS &
  PIDS="$PIDS $!"
fi
if [ "$TASK" = "all" ] || [ "$TASK" = "ppe" ]; then
  $CONDA_PY -m dog_ai_detection.helmet_vest_detection_node --ros-args \
    -r __node:=helmet_vest_detection_node --params-file $PARAMS $GUI_ARGS &
  PIDS="$PIDS $!"
fi

# RViz2
if [ $NO_RVIZ -eq 0 ]; then
  ros2 run rviz2 rviz2 -d $RVZ_CFG &
  PIDS="$PIDS $!"
fi

echo "=== 已启动, Ctrl+C 停止 ==="
trap 'kill $PIDS 2>/dev/null' INT TERM
wait
