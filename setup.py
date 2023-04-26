from setuptools import find_packages, setup

setup(
    name="mdel",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True
)
