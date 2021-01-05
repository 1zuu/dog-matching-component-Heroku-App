import os
num_classes = 5
seed = 42

width = 224
height = 224
n_channels = 3
target_size = (width, height)
input_shape = (width, height, n_channels)
rescale = 1./255

train_dir = 'data/Images/'
save_path = 'weights/numpy_images.npz'
cnn_converter_path = "weights/cnn_model.tflite"

csv_path = 'data/dog_reviews.csv'
max_length = 30
trunc_type = 'post'

tokenizer_path = 'weights/tokenizer.pickle'
rcn_converter_path = "weights/rcn_model.tflite"

##Inference
host = '0.0.0.0'
port = 5000
inference_save_path = 'weights/inference_images.npz'
dog_classes = {
            'shih tzu' : '0', 
            'papillon' : '1', 
            'maltese' : '2', 
            'afghan hound' : '3', 
            'beagle' : '4'
            }
n_neighbour_weights = 'weights/nearest neighbor weight folder/nearest neighbour {}.pkl'
n_neighbour = 3
min_test_sample = 30

cloud_image_dir = "https://res.cloudinary.com/douc1omvg/image/upload/Dog_Matching_Component/Images/"