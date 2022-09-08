# CNN-based wrist cartilage segmentation

This project contains sources code for the segmentation of wrist cartilage on MR-images for automatic volume quantification [[1]].

Examples of segmentation from [[1]]:
![examples](img/FIGURE_EXAMPLES2.svg?raw=true "Examples")


# Project structure:

[comment]: <> (- `./input` - folder containing data for training/testing )

[comment]: <> (- `./models` - folder containing model's weights)

[comment]: <> (- `./notebooks` - folder containing notebooks)
- `./configs` - folder containing configs in .yaml format
- `./wrist_segmentation` - folder containing source files
    - `./wrist_segmentation/models` - sources of models
    - `./wrist_segmentation/data` - sources of data manipulating
    - `./wrist_segmentation/utils` - sources of utilities scripts
- `./output` - folder containing output files e.g. plots etc

# Installation

[comment]: <> (TODO: upadting and etc)
In order to install the project you should have installed pip and use the following command inside the repo folder:
```
pip install -e .
```
It will install the project with requirements.
It recommended installing it in a separated virtual env or a conda env.


[1]: https://arxiv.org/abs/2206.11127