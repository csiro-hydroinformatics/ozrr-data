# OZ-DATA from python

[![license](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/csiro-hydroinformatics/camels-aus-py/blob/master/LICENSE) ![status](https://img.shields.io/badge/status-alpha-orange.svg) 

_This is currently a preview. You can contribute to features and design_

<!-- master: [![Build status - master](https://ci.appveyor.com/api/projects/status/vmwq7xarxxj8s564/branch/master?svg=true)](https://ci.appveyor.com/project/jmp75/camels-aus-py/branch/master) testing: [![Build status - devel](https://ci.appveyor.com/api/projects/status/vmwq7xarxxj8s564/branch/testing?svg=true)](https://ci.appveyor.com/project/jmp75/camels-aus-py/branch/testing) -->

**Python package to easily load and use the OZ-DATA dataset**

[Lerat, J., Thyer, M., McInerney, D., Kavetski, D., Woldemeskel, F., Pickett-Heaps, C., Shin, D., & Feikema, P. (2020). A robust approach for calibrating a daily rainfall-runoff model to monthly streamflow data. Journal of Hydrology, 591. https://doi.org/10.1016/j.jhydrol.2020.125129](https://doi.org/10.1016/j.jhydrol.2020.125129)

![Loading OZ-DATA from a notebook](./img/rapid_camels_load.png "Loading OZ-DATA from a notebook")

## License

BSD-3 (see [License](https://github.com/csiro-hydroinformatics/camels-aus-py/blob/master/LICENSE))

## Source code

The code repository is on [GitHub](https://github.com/csiro-hydroinformatics/camels-aus-py).

## Installation

### Linux

Using a conda environment is recommended. To create a new environment:

```bash
cd ${HOME}/tmp
wget https://raw.githubusercontent.com/csiro-hydroinformatics/camels-aus-py/main/configs/ozrr_environment.yml
my_env_name=camels
conda env create -n $my_env_name -f ./ozrr_environment.yml
conda activate $my_env_name 
```

Then:

```sh
pip install ozrr
```

If installing from source, after checking out this git repo:

```sh
pip install -r requirements.txt # if not using conda
python setup.py install
```

Developers:

```sh
python setup.py develop
```

### optional: setting jupyter-lab

optional but recommended: use mamba as a replacement for conda: `conda install -c conda-forge --name ${my_env_name} mamba`

```sh
mamba install -c conda-forge jupyterlab ipywidgets jupyter ipyleaflet
python -m ipykernel install --user --name ${my_env_name} --display-name "CAMELS"
jupyter-lab .
```

## Troubleshooting

### Notebooks

Normally jupyter-lab version 3.0 and more does not require explicit extensions installation, but if you have issues:

if: "Loading widgets..."

```sh
jupyter-labextension install @jupyter-widgets/jupyterlab-manager
```

if: "Error displaying widget: model not found"

```sh
jupyter-labextension install @jupyter-widgets/jupyterlab-manager
```

