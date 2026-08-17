cmake_minimum_required(VERSION 3.8)
project(dog_bringup)

find_package(ament_cmake REQUIRED)

# 将 config 安装到 share 目录
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME})

# 启动脚本
install(PROGRAMS
  scripts/start_detection.sh
  DESTINATION lib/${PROJECT_NAME})

ament_package()
