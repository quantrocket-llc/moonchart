#!/bin/bash -e

# PYPI_USERNAME - (Requried) Username for the publisher's account on PyPI
# PYPI_PASSWORD - (Required, Secret) Password for the publisher's account on PyPI

cat <<EOF >> ~/.pypirc
[distutils]
index-servers=pypi

[pypi]
username=$PYPI_USERNAME
password=$PYPI_PASSWORD
EOF

# Deploy to pip
python setup.py sdist bdist_wheel
twine upload dist/*

# Rebuild quantrocker/jupyter Docker image with latest package
curl -X POST https://registry.hub.docker.com/u/quantrocket/jupyter/trigger/41f6af9a-16bd-47c7-a088-71076407a7cc/
