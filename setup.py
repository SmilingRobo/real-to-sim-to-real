# setup.py

from setuptools import setup, find_packages

setup(
    name="real-to-sim-to-real",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pyyaml",
        "isaacgym"
    ],
    description="Framework for training and deploying real-to-sim-to-real robotic policies.",
    author="SmilingRobo",
)
