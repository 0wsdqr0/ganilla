# ganilla
style transform based on GAN


# how to use  
The data should be put at the root of the project. The structure of data should look like this: monet2photo/train/illustration/trainA, monet2photo/test/illustration/testA, monet2photo/train/photo/trainB, monet2photo/test/photo/testB.  

As long as the data is prepared, you can run python train.py to train from scratch.  If you have pre-trained parameters, make a directory in the root name {checkpoint}, then use python train.py --checkpoint_path {checkpoint} to train from the checkpoint. Remember to change variable epoch_choose at the start of the code to choose the epoch to start with.


You can also use python train.py --checkpoint_path {checkpoint} --test to generate style transformed photo with the test data.