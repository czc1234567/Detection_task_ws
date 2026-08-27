#!/usr/bin/env bash
# =====================================================================
# 巡检检测一键启动 (支持 4 类任务、2D 跟踪消抖与独立可视化控制)
#
# 单任务/组合任务启动:
#   bash start_detection.sh parking          # 仅车辆违规停放 + 行人
#   bash start_detection.sh ppe              # 仅安全帽 / 安全服
#   bash start_detection.sh fire             # 仅火焰检测
#   bash start_detection.sh smoke            # 仅吸烟检测
#   bash start_detection.sh all              # 启动全部 4 个模型
#
# 常用调试与显示选项:
#   bash start_detection.sh fire --gui       # 火焰任务 + 弹出本地 OpenCV 窗口
#   bash start_detection.sh all --no-rviz    # 全部任务 + 不打开 RViz2
#
# 性能优化 / 纯业务运行:
#   bash start_detection.sh fire --no-rviz --no-markers
# =====================================================================

set -e

TASK=all
NO_RVIZ=0
SHOW_GUI=0
PUB_RVZ_MARKERS=1

# 解析所有传入参数
for a in "$@"; do
  case "$a" in
    --no-rviz) NO_RVIZ=1 ;;
    --gui) SHOW_GUI=1 ;;
    --no-markers) PUB_RVZ_MARKERS=0 ;;
    parking|ppe|fire|smoke|all) TASK=$a ;;
  esac
done

CONDA_PY=/home/jetson/anaconda3/envs/yolo26/bin/python3
WS=/home/jetson/Detection_task_ws
PARAMS=$WS/install/dog_bringup/share/dog_bringup/config/params.yaml
RVZ_CFG=$WS/install/dog_bringup/share/dog_bringup/config/rviz_detection.rviz

# 环境源导入
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash

# 显式指定本地屏幕输出，确保 Qt/OpenCV 窗口与 RViz 能挂载到桌面
export DISPLAY=:0

# 布尔值转换 (显式覆盖 params.yaml 中定义的默认参数)
GUI_BOOL=$([ $SHOW_GUI -eq 1 ] && echo true || echo false)
MARKER_BOOL=$([ $PUB_RVZ_MARKERS -eq 1 ] && echo true || echo false)

echo "=== 检测启动 (task=$TASK, gui=$SHOW_GUI, rviz=$([ $NO_RVIZ -eq 0 ] && echo on || echo off), markers=$PUB_RVZ_MARKERS) ==="

# 1. 静态 TF 发布 (补齐 map -> camera_link 以及 camera_link -> camera_color_optical_frame 坐标链)
ros2 run tf2_ros static_transform_publisher \
  0 0 0 0 0 0 map camera_link > /dev/null 2>&1 &
PIDS=$!

ros2 run tf2_ros static_transform_publisher \
  0 0 0 -1.5707963 0 -1.5707963 camera_link camera_color_optical_frame > /dev/null 2>&1 &
PIDS="$PIDS $!"

# 任务 1: 违规停车 (绑定在第 4、5 核)
if [ "$TASK" = "all" ] || [ "$TASK" = "parking" ]; then
  taskset -c 4,5 $CONDA_PY -m dog_ai_detection.parking_detection_node --ros-args \
    -r __node:=parking_detection_node \
    --params-file $PARAMS \
    -p parking_detection_node:show_gui:=$GUI_BOOL \
    -p parking_detection_node:publish_rviz_markers:=$MARKER_BOOL &
  PIDS="$PIDS $!"
fi

# 任务 2: 安全帽/工装 (绑定在第 5、6 核)
if [ "$TASK" = "all" ] || [ "$TASK" = "ppe" ]; then
  taskset -c 5,6 $CONDA_PY -m dog_ai_detection.helmet_vest_detection_node --ros-args \
    -r __node:=helmet_vest_detection_node \
    --params-file $PARAMS \
    -p helmet_vest_detection_node:show_gui:=$GUI_BOOL \
    -p helmet_vest_detection_node:publish_rviz_markers:=$MARKER_BOOL &
  PIDS="$PIDS $!"
fi

# 任务 3: 火焰检测 (绑定在第 6、7 核)
if [ "$TASK" = "all" ] || [ "$TASK" = "fire" ]; then
  taskset -c 6,7 $CONDA_PY -m dog_ai_detection.fire_detection_node --ros-args \
    -r __node:=fire_detection_node \
    --params-file $PARAMS \
    -p fire_detection_node:show_gui:=$GUI_BOOL \
    -p fire_detection_node:publish_rviz_markers:=$MARKER_BOOL &
  PIDS="$PIDS $!"
fi

# 任务 4: 吸烟检测 (绑定在第 4、7 核)
if [ "$TASK" = "all" ] || [ "$TASK" = "smoke" ]; then
  taskset -c 4,7 $CONDA_PY -m dog_ai_detection.smoke_detection_node --ros-args \
    -r __node:=smoke_detection_node \
    --params-file $PARAMS \
    -p smoke_detection_node:show_gui:=$GUI_BOOL \
    -p smoke_detection_node:publish_rviz_markers:=$MARKER_BOOL &
  PIDS="$PIDS $!"
fi

# RViz2 可选启动
if [ $NO_RVIZ -eq 0 ]; then
  ros2 run rviz2 rviz2 -d $RVZ_CFG &
  PIDS="$PIDS $!"
fi

echo "=== 已启动, Ctrl+C 停止 ==="
trap 'kill $PIDS 2>/dev/null' INT TERM
wait