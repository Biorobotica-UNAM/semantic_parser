from setuptools import setup
from glob import glob
import os

package_name = "semantic_parser"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        # Registro del paquete
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # package.xml
        ("share/" + package_name, ["package.xml"]),
        # Launch files
        ("share/" + package_name + "/launch", glob("launch/*.launch.xml")),
        # JSON y otros recursos
        ("share/" + package_name + "/" + package_name, glob(f"{package_name}/*.json")),
    ],
    install_requires=["setuptools", "httpx"],
    zip_safe=True,
    maintainer="fidel",
    maintainer_email="jorge.martinezram@ingenieria.unam.edu",
    description="ROS2 service for Conceptual Dependency parsing via Ollama",
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "semantic_node = semantic_parser.semantic_node:main",
            "entity_extractor_node = semantic_parser.entity_extractor_node:main",
        ],
    },
)
