# LSTM WGAN-GP for trajectory generation

This repository contains the code for my bachelor thesis on synthetic generation of trajectory data.

## Steps to startup
1. Run ```pip install -r requirements.txt``` in a Python 3.8 environment
2. The main code for training the model lies in the ```lstmwgan.py``` file. Simply run it to start training.
   Alternatively you can also run the ```lstmwgan.ipynb``` notebook on google collab. 
   Already preprocessed sample data is provided in the ```data/preprocessed``` folder.
3. To generate trajectories run the ```predict.py``` file and change the ``n_epochs`` to the corresponding epoch number 
of the weights of the trained model. This will create an html file which shows the generated trajectories on a map.