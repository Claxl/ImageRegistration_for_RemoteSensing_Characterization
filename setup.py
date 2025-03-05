#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="remote_sensing_registration",
    version="1.0.0",
    description="A framework for evaluating image registration methods for remote sensing",
    author="Claudio Di Salvo",
    author_email="claudio.disalvo@mail.polimi.it",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'rs-register=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)