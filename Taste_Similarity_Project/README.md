***Taste similarity recognizer***

Funcitonality :

Given three images A, B, and C we find out if the food in image a tastes more similar to food in B or C. 

Methodology :

We use embeddings* as an input to our torch Neural Network, which consists of multiple Perceptron layers and different activaiton functions. The output of this represents the probability of A being more similar to B than C. 

Self-supervised :

We generate labels from our dataset by creating positive similarity rating to A, B, C if a is more similar to B than C, and a corresponding negative values wiith A, C, B. Or vice versa if A is more similar to C than B. 


Dataset :
The dataset consists of images in the food folder and their similarities are in the train.csv file




* (created by removing last layer of the pretrained Image recognigtion model ResNet152)
