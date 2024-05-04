from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name='exper',
    version='0.2.0',
    author="Mrzz",
    description="This is a python package for running deep learning experiments. "
                "Users can rapidly run their experiments by importing this module.",
    packages=find_packages(),
    long_description=long_description,
    install_requires=["torch", "torch_geometric"],
    include_package_data=True
)
