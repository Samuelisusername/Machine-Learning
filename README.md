**_Task 0:_**

Least Squares


**_Task 1:_**

Feature transformation and linear regression fitting for data analysis tasks. The transform_data function converts input features into 21 new features, incorporating linear, quadratic, exponential, and cosine transformations. Meanwhile, the fit function handles the training data points, transforms them, and fits linear regression on the transformed data, outputting optimal parameters. Explore efficient data transformation and regression fitting for your analytical workflows.

**_Task image recognition:_**

Analyzes taste based on images using ML. 
This project leverages a pretrained ResNet152 model to extract deep feature embeddings from images and trains a simple neural network to classify image triplets. It includes the following key features:

	•	Pretrained ResNet152: Downloads and uses ResNet152 to generate image embeddings.
	•	Triplet Data Handling: Loads and processes triplet data (anchor, positive, negative) for training.
	•	Custom Classifier: A simple neural network model for classifying image triplets.
	•	Training & Testing Pipelines: End-to-end pipeline for training the model on embeddings and testing its performance.
