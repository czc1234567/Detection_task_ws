# 巡检检测系统 (ROS 2 Humble)

奥比中光 Gemini 335L 相机 + YOLO 检测 + RViz/OpenCV 可视化。
**全部使用标准消息, 无自定义消息。**

## 架构 (便于后续叠加检测任务)

```
dog_ai_detection/
├── detection_core.py                 # 共享核心: 推理引擎 + 可视化节点基类
├── parking_detection_node.py         # 任务 1: 车辆违规停放 (+ 行人测试)
├── helmet_vest_detection_node.py     # 任务 2: 安全帽 / 安全服
└── models/                           # yolo26n.*(COCO), helmet.pt(PPE)

dog_bringup/
├── config/params.yaml                # 集中参数
├── config/rviz_detection.rviz        # RViz 配置
└── scripts/start_detection.sh        # 一键启动
```

新增检测任务 = 新增一个 `xxx_detection_node.py` 子类文件 + 参数段 + 脚本一行,
共用 `detection_core` 的订阅/推理/3D/标记/窗口全套能力, 无需重复实现。
任务多到一定规模后, 再按"任务族"拆分为独立功能包。

## 任务与模型

| 节点 | 模型 | 检测 | 违规判定 |
| --- | --- | --- | --- |
| `parking_detection_node` | yolo26n (COCO) | 车辆 (car/truck/bus/motorcycle) + 行人 | 车辆中心落入禁停区域多边形 |
| `helmet_vest_detection_node` | helmet.pt (PPE) | Helm/Vest/No_Helm/No_Vest | 模型直接输出 No_Helm/No_Vest |

> helmet.engine 与当前 ultralytics 版本不兼容, 默认自动使用 helmet.pt。
> 换用新模型时, 放到 `dog_ai_detection/models/` 并在对应节点的
> `MODEL_CANDIDATES` 中调整即可。

## 构建

```bash
cd ~/Detection_task_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 启动

```bash
# 一键启动 (RViz + 检测, AI 节点自动使用 conda yolo26 环境)
bash install/dog_bringup/lib/dog_bringup/start_detection.sh              # 全部
bash install/dog_bringup/lib/dog_bringup/start_detection.sh parking      # 仅停车
bash install/dog_bringup/lib/dog_bringup/start_detection.sh ppe          # 仅安全帽
bash install/dog_bringup/lib/dog_bringup/start_detection.sh parking --gui  # OpenCV 窗口
bash install/dog_bringup/lib/dog_bringup/start_detection.sh --no-rviz    # 不开 RViz
```

RViz 中可见: 目标 3D 标记 / 禁停区域 / 检测画面; 无地图时
`map` 与 `camera_link` 重合 (静态 TF), 3D 坐标即相机坐标系坐标。

## 话题 (全部标准消息)

- 订阅: `/camera/color/image_raw` `/camera/depth/image_raw` `/camera/color/camera_info`
- 发布: `/detection/result_image` (停车画面), `/detection/ppe_result_image` (PPE 画面),
  `/detection/object_markers` (3D 标记), `/detection/map_markers` (禁停区域),
  `/detection/alert` (Bool), `/detection/web_json` (String JSON)

## 环境说明

- AI 节点需 conda 环境 `yolo26` (torch 2.8 + ultralytics + tensorrt),
  脚本已自动使用 `/home/jetson/anaconda3/envs/yolo26/bin/python3`
- 禁停区域在 `config/params.yaml` 的 `parking_zone` / `fire_lane_zone`
  中用 map 坐标标定 (扁平坐标 [x1,y1,x2,y2,...])

