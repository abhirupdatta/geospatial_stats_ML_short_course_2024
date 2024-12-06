---
layout: page
title: Dependency installation 
description: >-
    Installation instruction.
---

# Installation and Setup Instructions

This guide outlines the steps to install and set up the necessary tools and packages for the short course, including R, Python, and `rpy2` for connecting Python to R.

------------------------------------------------------------------------

## Step 1: Install Conda

### 1.1 Install Miniconda or Anaconda

1.  Visit the [official Conda website](https://docs.conda.io/en/latest/miniconda.html).
2.  Download the installer for Miniconda (lightweight) or Anaconda (full distribution) appropriate for your operating system.
3.  Follow the installation instructions for your operating system.

------------------------------------------------------------------------

## Step 2: Create a Conda Environment

1.  Open your terminal (or command prompt on Windows).
2.  Create a new Conda environment with Python 3.10 by running:

``` bash
conda create -n my_env python=3.10
```

3.  Activate the new environment:

``` bash
conda activate my_env
```

------------------------------------------------------------------------

## Step 3: Install R and Required R Packages

### 3.1 Install R

1.  Go to the [R Project website](https://cran.r-project.org/).
2.  Download the appropriate version of R for your operating system (Windows, macOS, or Linux).
3.  Follow the installation instructions to complete the setup.

### 3.2 Install R Packages

Open an R session and run the following commands to install the required packages:

``` r
install.packages("tidyverse")
install.packages("geoR")
install.packages("BRISC")
install.packages("spNNGP")
install.packages("RandomForestGLS")
```

### 3.3. **Locate Your R Installation Path**

Copy the R_HOME directory (/path/to/R) returned by this function in R.

``` r
R.home()
```

### 3.4 **Set the `R_HOME` Environment Variable**

**On macOS/Linux**: Add the following line to your `.bashrc`, `.zshrc`, or equivalent shell configuration file:

``` bash
export R_HOME=/path/to/R
```

Then, apply the changes:

``` bash
source ~/.bashrc  # or ~/.zshrc
```

**On Windows** (Command Prompt):

``` cmd
set R_HOME=C:\path\to\R
```

------------------------------------------------------------------------

## Step 4: Install GeospaNN

### 4.1 Manual dependency installation

We provide options to install PyG libraries using conda and pip.

#### Option 1: Using Conda

For conda, installation in the following order is recommended. It may take around 10 minutes for conda to solve the environment for pytorch-sparse. The following chunk has been tested in a python 3.10 environment.

    #bash
    conda install pytorch torchvision -c pytorch
    conda install pyg -c pyg        
    conda install pytorch-sparse -c pyg 

#### Option 2: Using pip

For pip, installation in the following order is recommended to avoid any compilation issue. It may take around 15 minutes to finish the installation. The following chunk has been tested in a python 3.10 environment.

``` bash
pip install numpy==1.26 --no-cache-dir
pip install torch==2.0.0 --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0.html --no-cache-dir
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0.html --no-cache-dir
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0.html --no-cache-dir
pip install torch_geometric --no-cache-dir
```

### 4.2 Main installation

Once PyTorch and PyG are successfully installed, use the following command in the terminal for the latest version (version 11/2024):

    pip install https://github.com/WentaoZhan1998/geospaNN/archive/main.zip

To install the pypi version, use the following command in the terminal (version 1/2024):

    pip install geospaNN
