# Image Cartooning System using Machine Learning
# Complete implementation with preprocessing, model definition, training and inference

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from sklearn.model_selection import train_test_split
import tensorflow as tf
from glob import glob
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class ImageCartoonizer:
    def __init__(self, img_width=256, img_height=256):
        self.img_width = img_width
        self.img_height = img_height
        self.model = None
        
    def build_generator(self):
        """
        Build the U-Net style generator for image-to-cartoon transformation
        """
        def conv_block(input_tensor, num_filters, kernel_size=3, strides=1, activation='relu', batch_norm=True):
            x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
            if batch_norm:
                x = BatchNormalization()(x)
            if activation:
                x = Activation(activation)(x)
            return x
        
        # Input
        inputs = Input(shape=(self.img_height, self.img_width, 3))
        
        # Encoder (downsampling)
        e1 = conv_block(inputs, 64, kernel_size=7)
        e2 = conv_block(e1, 64, strides=2)
        e2 = conv_block(e2, 128, kernel_size=3)
        e3 = conv_block(e2, 128, strides=2)
        e3 = conv_block(e3, 256, kernel_size=3)
        
        # Residual blocks
        r = e3
        for i in range(6):  # 6 residual blocks
            r_temp = conv_block(r, 256, kernel_size=3)
            r_temp = conv_block(r_temp, 256, kernel_size=3, activation=None)
            r = Add()([r, r_temp])
        
        # Decoder (upsampling)
        d1 = UpSampling2D(size=2)(r)
        d1 = conv_block(d1, 128, kernel_size=3)
        d2 = UpSampling2D(size=2)(d1)
        d2 = conv_block(d2, 64, kernel_size=3)
        
        # Output
        outputs = Conv2D(3, kernel_size=7, padding='same', activation='tanh')(d2)
        outputs = (outputs + 1) / 2  # Scale from [-1, 1] to [0, 1]
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def prepare_dataset(self, photo_dir, cartoon_dir, batch_size=8, split=0.2):
        """
        Prepare dataset for training
        photo_dir: directory containing real photos
        cartoon_dir: directory containing cartoon images
        """
        photo_paths = glob(os.path.join(photo_dir, "*.jpg")) + glob(os.path.join(photo_dir, "*.png"))
        cartoon_paths = glob(os.path.join(cartoon_dir, "*.jpg")) + glob(os.path.join(cartoon_dir, "*.png"))
        
        # Take the minimum length to ensure balanced dataset
        min_len = min(len(photo_paths), len(cartoon_paths))
        photo_paths = photo_paths[:min_len]
        cartoon_paths = cartoon_paths[:min_len]
        
        # Split into train and validation
        train_photos, val_photos = train_test_split(photo_paths, test_size=split, random_state=42)
        train_cartoons, val_cartoons = train_test_split(cartoon_paths, test_size=split, random_state=42)
        
        # Create TensorFlow datasets
        def load_and_preprocess(img_path, is_cartoon=False):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [self.img_height, self.img_width])
            img = tf.cast(img, tf.float32) / 127.5 - 1  # Normalize to [-1, 1]
            return img
        
        # Create datasets
        train_photos_ds = tf.data.Dataset.from_tensor_slices(train_photos).map(
            lambda x: load_and_preprocess(x, False))
        train_cartoons_ds = tf.data.Dataset.from_tensor_slices(train_cartoons).map(
            lambda x: load_and_preprocess(x, True))
        val_photos_ds = tf.data.Dataset.from_tensor_slices(val_photos).map(
            lambda x: load_and_preprocess(x, False))
        val_cartoons_ds = tf.data.Dataset.from_tensor_slices(val_cartoons).map(
            lambda x: load_and_preprocess(x, True))
        
        # Pair images
        train_ds = tf.data.Dataset.zip((train_photos_ds, train_cartoons_ds))
        val_ds = tf.data.Dataset.zip((val_photos_ds, val_cartoons_ds))
        
        # Batch and prefetch
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def build_vgg_feature_network(self):
        """
        Build a VGG19 model for feature extraction (for perceptual loss)
        """
        vgg = VGG19(include_top=False, weights='imagenet', 
                    input_shape=(self.img_height, self.img_width, 3))
        vgg.trainable = False
        
        output_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
        outputs = [vgg.get_layer(name).output for name in output_layers]
        
        model = Model(inputs=vgg.input, outputs=outputs)
        return model
    
    def perceptual_loss(self, y_true, y_pred):
        """
        Calculate perceptual loss using VGG19 features
        """
        vgg = self.build_vgg_feature_network()
        
        # Preprocess images before VGG
        y_true = (y_true + 1) * 127.5  # Scale back to [0, 255]
        y_pred = (y_pred + 1) * 127.5  # Scale back to [0, 255]
        
        # Get features
        true_features = vgg(y_true)
        pred_features = vgg(y_pred)
        
        # Calculate L1 loss for each feature layer
        loss = 0
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss += tf.reduce_mean(tf.abs(true_feat - pred_feat))
        
        return loss
    
    def combined_loss(self, y_true, y_pred):
        """
        Combined loss function with L1 and perceptual loss
        """
        # L1 loss (pixel-wise)
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # Perceptual loss
        p_loss = self.perceptual_loss(y_true, y_pred)
        
        # Return combined loss
        return l1_loss + 0.1 * p_loss
    
    def train(self, train_ds, val_ds, epochs=50):
        """
        Train the cartooning model
        """
        # Build generator model
        self.model = self.build_generator()
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=2e-4),
            loss='mae'  # Mean Absolute Error (L1)
        )
        
        # Train model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='cartoon_generator_model.h5',
                    monitor='val_loss',
                    save_best_only=True
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        return history
    
    def load_model(self, model_path):
        """
        Load a pre-trained model
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        return self.model
    
    def cartoonize(self, input_img_path, output_img_path=None):
        """
        Convert an image to cartoon style
        input_img_path: path to input image
        output_img_path: path to save output image (optional)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Either train a model or load a pre-trained one.")
        
        # Load and preprocess input image
        img = cv2.imread(input_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Resize for model
        input_img = cv2.resize(img, (self.img_width, self.img_height))
        input_img = input_img.astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1]
        
        # Generate cartoon
        cartoon = self.model.predict(np.expand_dims(input_img, axis=0))[0]
        cartoon = (cartoon * 255).astype(np.uint8)  # Scale to [0, 255]
        
        # Resize back to original dimensions
        cartoon = cv2.resize(cartoon, (w, h))
        
        # Save if output path is provided
        if output_img_path:
            plt.imsave(output_img_path, cartoon)
        
        return cartoon
    
    def cartoonize_with_edge_enhancement(self, input_img_path, output_img_path=None):
        """
        Enhanced version that combines ML cartooning with edge detection
        """
        # Generate base cartoon using the ML model
        cartoon_img = self.cartoonize(input_img_path)
        
        # Load original image for edge detection
        original_img = cv2.imread(input_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Edge detection
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 5)
        
        # Convert edges to 3 channels
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Combine cartoon with edges
        cartoon_with_edges = cv2.bitwise_and(cartoon_img, edges)
        
        # Save if output path is provided
        if output_img_path:
            plt.imsave(output_img_path, cartoon_with_edges)
        
        return cartoon_with_edges

# Usage example for this image cartooning system
def main():
    # Initialize the cartoonizer
    cartoonizer = ImageCartoonizer(img_width=256, img_height=256)
    
    # If you have a dataset and want to train the model:
    """
    # Prepare dataset
    train_ds, val_ds = cartoonizer.prepare_dataset(
        photo_dir='path/to/real/photos',
        cartoon_dir='path/to/cartoon/images',
        batch_size=8
    )
    
    # Train model
    history = cartoonizer.train(train_ds, val_ds, epochs=50)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    """
    
    # Load a pre-trained model (if you have one)
    # cartoonizer.load_model('cartoon_generator_model.h5')
    
    # Alternative: If you don't have a model or dataset, you can use a simplified approach
    def simplified_cartoonize(input_path, output_path):
        # Read image
        img = cv2.imread(input_path)
        
        # Convert to RGB (for display) and grayscale (for edge detection)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for smoothing while preserving edges
        img_blur = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Edge detection
        img_edge = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, blockSize=9, C=2)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        
        # Combine edge with blurred image
        cartoon = cv2.bitwise_and(img_blur, img_edge)
        
        # Save result
        plt.imsave(output_path, cartoon)
        return cartoon
    
    # Example usage of simplified approach (doesn't require ML model)
    # simplified_cartoonize('input.jpg', 'cartoon_output.jpg')
    
    # For real ML-based approach, use:
    # cartoonizer.cartoonize('input.jpg', 'cartoon_output.jpg')
    # or
    # cartoonizer.cartoonize_with_edge_enhancement('input.jpg', 'cartoon_output_enhanced.jpg')

if __name__ == "__main__":
    main()
