================================================================================

FOUNDATIONS OF COMPUTER VISION : FINAL PROJECT

ACTION RECOGNITION USING 3D CONVOLUTIONAL NETWORKS

================================================================================

=======
AUTHORS
=======
==> DELZAD BAMJI (db3765@rit.edu)
==> SIDDARTH SARGUNARAJ (sxs2469@rit.edu)


1] INSTRUCTIONS:
   --> Place the dataset in the data folder
       --> Our implementation used the dataset found at 
           https://drive.google.com/drive/u/1/folders/1J-IYgzyNfiR33DazQjYioKEuSt-dHEq7
       --> To use a custom dataset, place it in the data folder. 
           Set the root_dir variable in mypath.py so that it points to your custom dataset
   --> Using different models
       --> We have implemented two models, C3D and R2Plus1D
       --> To use C3D, set the modelName variable in train.py and test.py to C3D
       --> To use R2Plus1D, set the modelName variable in train.py and test.py to R2Plus1D
   --> Place the testing video in the assets folder
   --> Graphs can be created by running tensorboard --logdir=$EVENTS_FOLDER
       --> The EVENTS_FOLDER is found in under /run_*/models
       --> It contains the tf.event file
       --> The graphs can viewed in a web browser at http://localhost:6006      

2] TROUBLESHOOTING
   --> If test.py cannot find the *.pth.tar file, 
       ensure that the path in test.py is set to the latest run_* folder           
       
================================================================================
