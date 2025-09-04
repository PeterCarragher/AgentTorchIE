"""
Temporary setup.py for editable installation.
This reads from pyproject.toml for package metadata.
"""

from setuptools import setup, find_packages
import tomli

# Read pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)

project_config = pyproject["project"]

setup(
    name=project_config["name"],
    version=project_config["version"],
    description=project_config["description"],
    author=project_config["authors"][0]["name"],
    packages=find_packages(),
    install_requires=project_config["dependencies"],
    python_requires=project_config["requires-python"],
    package_data={
        "agent_torch": ["**/*.yaml", "**/*.yml", "**/*.json"],
    },
    include_package_data=True,
)