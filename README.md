# CVProject

## Overview
This project is focused on developing a computer vision solution for fall detection using various deep learning models. It aims to enhance monitoring and safety for elderly care by leveraging models like CNN, DenseNet201, ResNet50, and others.

## Features
- Fall detection using different pre-trained models.
- Performance comparison between models.
- Confusion matrices for model evaluation.
- Jupyter Notebook for model training and analysis.

## Methodology

### Model Training
The training process involves an iterative approach to train multiple deep learning models using a consistent methodology. The models included in the study are DenseNet201, ResNet50, VGG19, InceptionResNetV2, and a custom Convolutional Neural Network (CNN). Each model's training process is defined as follows:

1. **Model Initialization**: Each model is instantiated using pre-defined functions that configure the architecture. For example, the `create_densenet201_model()` function loads the DenseNet201 architecture without its top layers and adds custom layers, including a global average pooling layer, a dense layer, a Multi-Head Self Attention layer, and a final output layer.

2. **Training Configuration**: Each model is compiled with an Adam optimizer and a custom focal loss function, which addresses class imbalance by applying a focusing parameter (\(\gamma\)) and a balancing parameter (\(\alpha\)). The models are trained on the training dataset (`X_train`, `y_train`) for a maximum of 20 epochs, with validation data (`X_val`, `y_val`) to monitor performance.

3. **Call Back Functions**: Early stopping is utilized to prevent overfitting, halting training if the validation accuracy does not improve for a set number of epochs (patience). Additionally, a learning rate reduction callback adjusts the learning rate based on validation loss, allowing for a more dynamic learning process.

4. **Model Training Iteration**: For each model in the `models_dict` dictionary, the training process is repeated:
   - The model is trained on the training data.
   - After training, the best weights are restored to ensure optimal model performance.
   - The model is saved to a specified path for future use.

5. **Result Storage**: After each model's training, performance metrics (accuracy, precision, recall, F1-score) are computed and stored in a results dictionary, allowing for easy comparison across different architectures.

### Evaluation Metrics
To comprehensively evaluate the performance of the trained models, several metrics were used:

1. **Accuracy**: Measures the proportion of correct predictions among the total predictions, providing a general sense of model performance.

2. **Precision**: Calculates the ratio of true positive predictions to the total number of predicted positives, indicating the reliability of positive predictions.

3. **Recall**: Measures the ratio of true positives to the actual number of positives, reflecting the model's ability to identify positive instances.

4. **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two, particularly useful for imbalanced datasets.

5. **Confusion Matrix**: A visual representation of the model's performance, showing the counts of true positive, false positive, true negative, and false negative predictions for a detailed examination.

## Files
- `fallDetection.ipynb`: Main code file for training and testing models.
- Confusion Matrix images (`confusion_matrix_*.png`): Visual representation of model performance.
- `Fall Detection - Comparison.jpg`: Summary comparison of model results.

## Requirements
- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- Jupyter Notebook

## How to Use
Clone the repository:
```bash
git clone https://github.com/AdityaKhandelwal2306/CVProject.git
```

## Results
Detailed confusion matrices are available for each model, highlighting accuracy and misclassifications. The project compares different models to determine the best-performing architecture for fall detection.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any suggestions or improvements.
