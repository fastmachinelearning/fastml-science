#!/bin/bash

echo "Environment setup script"

conda env create -f environment.yml
conda activate ecoder-env
