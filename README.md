<h1> Image Forgery Detection and Localization </h1>

<h2> Overview</h2>

This project focuses on detecting and localizing image forgeries using deep learning techniques. It employs convolutional neural networks (CNNs) for classification and U-Net for localization, leveraging transfer learning with EfficientNetV2.

<h2> Methodology</h2>

<h3> 1. Data Preparation </h3>
	
	Import datasets: CASIA v2.0, CoMoFoD
	
	Preprocess images using Error Level Analysis, Rescaling, and Normalization
	
	Split dataset: 80% training, 20% testing
	
	Randomly shuffle the dataset

<h3> 2. Image Forgery Classification </h3>

	Define CNN architecture using EfficientNetV2 (pre-trained on ImageNet)
	
	Modify the top layers for transfer learning
	
Train the model:
	
	Set epochs and batch size
	
	Load and preprocess dataset
	
	Fine-tune by freezing initial layers to prevent overfitting
	
	Evaluate performance:
	
	Calculate training/testing accuracy
	
	Plot loss curves
	
	Compute precision, recall, and F1-score

<h3> 3. Image Forgery Localization </h3>
	Implement U-Net for pixel-wise forgery detection
	
	Use EfficientNetV2 as an encoder with a U-Net decoder
	
	Incorporate skip connections and convolutional layers
	
Train the model:
	
	Set epochs and batch size
	
	Load and preprocess dataset
	
	Fine-tune using transfer learning
	
Evaluate performance:
	
	Calculate training/testing accuracy
	
	Plot loss curves
	
	Compute precision, recall, and F1-score
	
	Display performance metrics
	
	Results and Evaluation
	
	The model provides accurate forgery classification and localization.
	
	Performance metrics include accuracy, precision, recall, and F1-score.

<h2> Dependencies </h2>
	TensorFlow / PyTorch
	
	EfficientNetV2
	
	U-Net
	
	Image processing libraries
	
<h2> Usage</h2> 
	
	Prepare and preprocess the dataset.
	
	Train the classification model to detect forged images.
	
	Train the localization model to pinpoint forged areas.
	
	Evaluate model performance using precision, recall, and accuracy.
	
	Use the trained model to make predictions on test images.

<h2> Conclusion</h2> 

	This project effectively classifies and localizes image forgeries using deep learning techniques, leveraging EfficientNetV2 and U-Net for high accuracy and robustness.

Note: Ensure GPU acceleration for optimal performance during training.

