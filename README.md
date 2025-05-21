# Image-Caption-Generator-Using-Deep-Learning-Image-Captioning-Using-CNN-LSTM


kaggle link:-https://www.kaggle.com/code/manish29k/image-caption-generator




# Image Caption Generator

An AI-powered application that generates descriptive captions for images using deep learning techniques.

## Overview

This project implements an image captioning system that combines computer vision and natural language processing to automatically generate descriptive captions for images. The model uses a CNN (Convolutional Neural Network) for feature extraction from images and an LSTM (Long Short-Term Memory) network for generating natural language descriptions.

## Features

- **Automatic Image Captioning**: Generate descriptive captions for any uploaded image
- **User-friendly Interface**: Simple web interface built with Streamlit for easy image uploading and caption generation
- **Pre-trained Models**: Utilizes pre-trained feature extraction and caption generation models

## Dataset

The model was trained on the Flickr8k dataset, which contains 8,000 images, each with 5 different captions. This diverse dataset helps the model learn to generate accurate and contextually relevant descriptions for a wide variety of images.

## Model Architecture

The image captioning system consists of two main components:

1. **Feature Extraction**: A pre-trained CNN model (DenseNet201) that extracts high-level features from input images
2. **Caption Generation**: An LSTM-based decoder that generates captions word by word based on the extracted image features

## Installation

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Streamlit
- NumPy
- Matplotlib
- Pillow

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-caption-generator.git
   cd image-caption-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the pre-trained models:
   - Create a 'models' directory in the project root
   - Download the feature extractor model, caption model, and tokenizer from the release section
   - Place them in the 'models' directory

## Usage

### Running the Web Interface

1. Start the Streamlit app:
   ```
   streamlit run main.py
   ```

2. Open your web browser and navigate to http://localhost:8501

3. Upload an image using the file uploader

4. The application will process the image and display the generated caption

## How It Works

1. The uploaded image is processed and resized to the required dimensions
2. The feature extraction model extracts visual features from the image
3. These features are fed into the caption generation model
4. The caption generator produces a descriptive caption word by word
5. The final caption is displayed alongside the uploaded image

## Model Training

The notebook `image-caption-generator.ipynb` contains the complete code for training the model from scratch. The training process includes:

1. Loading and preprocessing the Flickr8k dataset
2. Extracting features from images using a pre-trained CNN
3. Preparing text data with tokenization and sequence padding
4. Building and training the caption generation model
5. Evaluating the model performance

## Project Structure

```
image-caption-generator/
├── main.py                    # Streamlit web application
├── image-caption-generator.ipynb  # Jupyter notebook with model training code
├── models/                    # Directory for pre-trained models
│   ├── model.keras            # Caption generation model
│   ├── feature_extractor.keras # Image feature extraction model
│   └── tokenizer.pkl          # Tokenizer for text processing
├── uploaded_image.jpg         # Temporary storage for uploaded images
└── README.md                  # Project documentation
```

## Future Improvements

- Implement beam search for better caption generation
- Train on larger datasets (MSCOCO, Flickr30k) for improved performance
- Add support for video captioning
- Implement attention mechanisms for more accurate captions
- Add multilingual caption generation

## Kaggle Notebook

For more details on the model training and implementation, check out the original Kaggle notebook:
[Image Caption Generator on Kaggle](https://www.kaggle.com/code/manish29k/image-caption-generator)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Flickr8k dataset for providing training data
- TensorFlow and Keras for deep learning frameworks
- Streamlit for the web interface
