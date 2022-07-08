# CycleGAN
Simple Python CycleGAN implementation from scratch in Pytorch.<br>
Project contains two configuration: train and generate section.<br>
Train configuration is used for training and evaluation of data.<br>
Generate configuration is used for generating images from pretrained models.<br>
## Train configuration:
`--train` <br>
`--generate_dir` [path to generated images]<br>
`--carla_dir` [path to source images]<br>
`--darwin_dir` [path to target images]<br>
optional:<br>
`--load_model` [path for checkpoints]<br>
`--save_model` [path to future check_points]<br>

## Generate configuration:
`--generate`<br>
`--load_model` [path for checkpoints]<br>
`--generate_dataset_dir` [path to dataset]<br>
`--generate_dir` [path to generated images]<br>

<br>


    Petar Stojkovic
    Deep Learning Engineer
    University of Belgrade, School of Electrical Engineering
