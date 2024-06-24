import numpy as np
import tensorflow as tf
from torchvision import datasets
from tensorflow.keras import models, layers
from PIL import Image

# Set the data directory
#DATA_DIR = "C:\\Users\\Karabo\\OneDrive - University of Cape Town\\CSC3022F\\Machine Learning\\LAB1\\MNIST_data"

# Load MNIST dataset
download_dataset = True
train_mnist = datasets.MNIST("", train=True, download=download_dataset)
test_mnist = datasets.MNIST("", train=False, download=download_dataset)

x_train, y_train = train_mnist.data.numpy(), train_mnist.targets.numpy()
x_test, y_test = test_mnist.data.numpy(), test_mnist.targets.numpy()

# Normalize data from 0-255 to 0-1 (important for training with TensorFlow)
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Function to log training progress
class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super(TrainingLogger, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            f.write(f"Epoch [{epoch+1}/{self.params['epochs']}], Loss: {logs['loss']}\n")

# Define log file path
LOG_FILE = "log.txt"

# Train the model with logging
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[TrainingLogger(LOG_FILE)])

# Log final accuracy
test_loss, test_accuracy = model.evaluate(x_test, y_test)
with open(LOG_FILE, 'a') as f:
    f.write(f"Final Test Loss: {test_loss}, Final Test Accuracy: {test_accuracy}\n")

# Function to preprocess user input image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.LANCZOS)  # Resize to 28x28 using LANCZOS
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0  # Normalize the image
    img_array = img_array.reshape(1, 28, 28)  # Reshape for the model
    return img_array

# Main function to handle user input
def main():
    while True:
        filepath = input("Please enter a filepath (type 'exit' to quit): ")
        if filepath.lower() == 'exit':
            print("Exiting...")
            break
        try:
            input_image = preprocess_image(filepath)
            prediction = model.predict(input_image)
            predicted_digit = np.argmax(prediction)
            print(f"Classifier: {predicted_digit}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
