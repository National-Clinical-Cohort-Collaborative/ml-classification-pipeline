#!/usr/bin/env python
"""Library setup script."""

import os
from setuptools import find_packages, setup

setup(
    name=os.environ['PKG_NAME'],
    version=os.environ['PKG_VERSION'],

    description='My Python library project',

    author="UNITE",

    packages=find_packages(exclude=['contrib', 'docs', 'test']),

    # Please instead specify your dependencies in conda_recipe/meta.yml
    install_requires=[],

    entry_points={
        'transforms.pipeline': [
            'root = prediction_pipelines.prediction_pipeline:prediction_pipeline'
        ]
    }
)
