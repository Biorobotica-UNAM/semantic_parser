from setuptools import find_namespace_packages, setup
from glob import glob

package_name = "semantic_parser"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_namespace_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        # Registro del paquete
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # package.xml
        ("share/" + package_name, ["package.xml"]),
        # Launch files
        ("share/" + package_name + "/launch", glob("launch/*.launch.xml")),
    ],
    install_requires=["setuptools", "httpx", "PyYAML"],
    zip_safe=True,
    maintainer="fidel",
    maintainer_email="jorge.martinezram@ingenieria.unam.edu",
    description="ROS2 services for Qwen goals and YAML entity extraction",
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "qwen_semantic_node = semantic_parser.qwen_semantic_node:main",
            "entity_extractor_node = semantic_parser.entity_extractor_node:main",
        ],
    },
)
