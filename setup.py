import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "analytics",
    version = "1.0.0",
    description = ("Autocomplete missing voxel in reconstructed MRSI data"),
    license = "BSD",
    keywords = "example documentation tutorial",
    packages=find_packages(include=['fillgaps','tools']),
    # long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
