# CarND-Behavioral-Cloning
My behavioral cloning project from Udactiy's Self-Driving Car Nanodegree.

## Data
I collected data by driving around both tracks normally for four laps each. I then collected "recovery" data for upwards of six laps by driving to the side of the track, then recording my movements to get back on a correct path. I went back and collected more data around tricky corners since most of the tracks were straight or soft curves.

I preprocessed the data for training by resizing to 32x16 pixel images and converting to grayscale, both for speed, and normalizing image values between -1 and 1. I used 10% of the total data as a validation set and the rest for training. I didn't create a different testing set because the real testing could be done by running the simulator in autonomous mode to get qualitative results.

## Model
For the model architecture I chose four convolutional layers to extract features from the camera images, followed by four fully-connected layers to . Each layer besides the last is activated by a ReLU, while the last layer is activated by tanh to keep the prediction angles between -1 and 1. I introduced two dropout layers - one after the first two convolutional layers and one after the second two - to reduce the possibility of overfitting. 

I originally started out with a more complex model with eight convolutional layers, but I found that this took much longer to train and didn't produce much better results, so I pruned down the model for simplicity and speed, while still keeping accuracy high. 

I optimized the model with an Adam optimizer over MSE loss.

## Hyperparameters
I chose a learning rate of 0.0001 rather than the default adam optimizer rate of 0.001 because I found that the higher rate plateaued at a higher loss and produced worse qualitative driving results. I trained for 10 epochs because performance increase started diminishing when training for longer.

## Results
I received a final training loss of 0.0514 and a validation loss of 0.0543. Qualitatively, the model drives the car well on both tracks (best performance at smallest resolution and lowest graphics), without ever crashing or venturing into dangerous areas.