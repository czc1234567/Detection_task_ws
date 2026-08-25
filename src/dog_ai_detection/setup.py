from setuptools import find_packages, setup

package_name = 'dog_ai_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    # 模型文件随 Python 包一起安装 (含 .pt / .engine / .onnx)
    package_data={package_name: ['models/*']},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dog Inspection Team',
    maintainer_email='dev@example.com',
    description='AI vision detection node (YOLO/TensorRT placeholder) for the dog inspection system.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'parking_detection_node = dog_ai_detection.parking_detection_node:main',
            'helmet_vest_detection_node = dog_ai_detection.helmet_vest_detection_node:main',
            'fire_detection_node = dog_ai_detection.fire_detection_node:main',
            'smoke_detection_node = dog_ai_detection.smoke_detection_node:main',
        ],
    },
)