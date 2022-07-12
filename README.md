# CNN-based wrist cartilage segmentation

This project contains sources code for the segmentation of wrist cartilage on MR-images for automatic volume quantification [[1]].

# Project structure:
- `./input` - folder containing data for testing 
- `./models` - folder containing model's weights
- `./`  
- `./wrist_segmentation` - folder containing source files
    - `./wrist_segmentation/models` - sources of models
    - `./wrist_segmentation/data` - sources of data manipulating
    - `./wrist_segmentation/train` - sources of training scripts
    - `./wrist_segmentation/test` - sources of test/inference scripts
    - `./wrist_segmentation/utils` - sources of utilities scripts
- `./output` - folder containing output files e.g. plots etc


# Author
`Nikita Vladimirov`

[1]: https://arxiv.org/abs/2206.11127