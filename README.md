# Transverse Velocity Field Measurement of Solar High-Resolution Images Based on Unsupervised Deep Learning
The project is to calculate the Transverse velocity field of the solar image and achieve accurate optical flow estimation by calculating the pixel displacement in the image. 

# This repository includes:
pretrained model and source code of our paper
# Requirements
The code has been tested with PyTorch 1.13 and Cuda 11.6.
```Shell
conda env create -f pwc1.yaml
conda activate pwc1
```

# Demos
Pretrained model can be find in
```Shell
./checkpoints/solar-un-ar-fbar-loss-s.pth
```

You can demo a trained model on a sequence of frames
```Shell
python .\demo_un.py --model ./checkpoints/solar-un-ar-fbar-loss-s.pth --path ./dataset/solar_demo/three_velocity/
```

# Training
After you have prepared the optical flow dataset, you can run train_pwcnet_ablation.py:
```Shell
python train_bf_un_ar_fbar_loss.py --name biflow_un --stage solar --validation solar --gpus 0 --num_steps 90000 -batch_size 2 --lr 0.001 --image_size 384 448 --wdecay 0.0001 --gamma=0.5
```
