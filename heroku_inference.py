import os
import time
import pickle
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.neighbors import NearestNeighbors
from util import load_inference_data, get_prediction_data, load_labeled_data, rescale_imgs
from variables import *

class KerasToTFConversion(object):
    def TFinterpreter(self, model_converter):
        self.interpreter = tf.lite.Interpreter(model_path=model_converter)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def Inference(self, inputs):
        input_shape = self.input_details[0]['shape']
        input_data = np.expand_dims(inputs, axis=0).astype(np.float32)
        
        assert np.array_equal(input_shape, input_data.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

class InferenceModel(object):
    def __init__(self):
        image_labels, inference_images, image_urls = load_inference_data()
        self.inference_images = inference_images
        self.image_labels = image_labels
        self.image_urls = image_urls

        rcn_inference = KerasToTFConversion()
        cnn_inference = KerasToTFConversion()
        cnn_inference.TFinterpreter(cnn_converter_path)
        rcn_inference.TFinterpreter(rcn_converter_path)

        self.rcn_inference = rcn_inference
        self.cnn_inference = cnn_inference
        
    def extract_image_features(self, label):

        self.image_labels_class, self.inference_images_class, self.image_urls_class = load_labeled_data(
                                                                                        self.image_labels, 
                                                                                        self.inference_images, 
                                                                                        self.image_urls,
                                                                                        label)
        if not os.path.exists(n_neighbour_weights.format(label)):
            self.test_features = np.array(
                            [self.cnn_inference.Inference(img) for img in self.inference_images_class]
                                        )
            self.test_features = self.test_features.reshape(self.test_features.shape[0],-1)
            self.neighbor = NearestNeighbors(
                                        n_neighbors = n_neighbour
                                        )
            self.neighbor.fit(self.test_features)
            with open(n_neighbour_weights.format(label), 'wb') as file:
                pickle.dump(self.neighbor, file)
        else:
            with open(n_neighbour_weights.format(label), 'rb') as file:
                self.neighbor = pickle.load(file)

    def extract_text_features(self, text_pad):
        return self.rcn_inference.Inference(text_pad)

    def predictions(self, text_pad, show_fig=False):
        n_neighbours = {}
        fig=plt.figure(figsize=(8, 8))
        text_pad = self.extract_text_features(text_pad)
        text_pad = text_pad.reshape(1, -1)
        result = self.neighbor.kneighbors(text_pad)[1].squeeze()
        for i in range(n_neighbour):
            neighbour_img_id = result[i]
            img = self.inference_images_class[neighbour_img_id]
            url = self.image_urls_class[neighbour_img_id]
            img = rescale_imgs(img)
            fig.add_subplot(1, 3, i+1)
            plt.title('Neighbour {}'.format(i+1))
            plt.imshow((img * 255).astype('uint8'))
            n_neighbours["Neighbour {}".format(i+1)] =  "{}".format(url)
        if show_fig:
            plt.show()
        return n_neighbours