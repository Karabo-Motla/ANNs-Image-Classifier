README for the MNIST Classifier Program

Overview
This Python-based program is designed to classify handwritten digits using the MNIST dataset. 
It leverages TensorFlow to build and train a neural network and includes custom functionality for
 logging training progress and predicting digits from user-provided images.

Files and Structure
classifier.py: This is the main script of the project. It includes the entire pipeline required for training the model,
 evaluating its performance, and making predictions on new images.

How to Run the Program
Adjust Data Directory: Before running the script, ensure you set the DATA_DIR variable in classifier.py 
to a valid path on your system where the MNIST data should be stored or is already stored.
eg  
>>DATA_DIR = "C:\\Users\\Karabo\\OneDrive - University of Cape Town\\CSC3022F\\Machine Learning\\LAB1\\MNIST_data"

Install Dependencies: Install the required libraries using pip:
>>pip install numpy tensorflow torch torchvision pillow

Execute the Script: Run classifier.py from the command line:
>>python classifier.py


The file paths in classifier.py are configured for the developer's environment. 
You must change the DATA_DIR to the corresponding directory on your machine.
When prompted for an image file path, ensure you provide a valid path relative to your
current directory or an absolute path.

Running the Program
The program will display training progress in the terminal, and once training is complete, 
you can enter an image path when prompted to get the classified digit.

Regarding Log Messages
During the execution of the program, you may notice some informational messages that appear in the console. 
These messages, such as those pertaining to oneDNN custom operations or CPU feature guards, are not errors. 
They are informative logs from TensorFlow indicating optimizations and instructions used.
The warnings about oneDNN and CPU instructions are harmless and do not affect the execution of your program. 
If you prefer not to see these logs, you can set the environment variable TF_ENABLE_ONEDNN_OPTS=0. 
However, this is not necessary for the functioning of the program.