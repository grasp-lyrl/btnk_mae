# created By: Zhaoze Wang
from setuptools import setup, find_packages

setup(
    name='btnk_mae',
    version='0.0.0',
    description='Bottlenecked Masked AutoEncoder for Compact Image Representations',
    author='Zhaoze Wang',
    license='MIT',
    
    packages=find_packages(include=['btnk_mae', 'btnk_mae.*']),
    python_requires='>=3.9',
)
