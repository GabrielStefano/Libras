from keras.preprocessing import image
# from keras.applications.resnet import ResNet152, preprocess_input
# from keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
# from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, Dense, Dropout, MaxPooling2D, Flatten
import cv2
import numpy

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=False, #True

            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        # if weights is None:
        #     # Get model with pretrained weights.
        #     self.model = VGG19(
        #         weights='imagenet',
        #         include_top=False,
        #         pooling='avg'
        #     )
            # cnn_out = GlobalAveragePooling2D()(self.model.output)
            # self.model = Model(input=self.model.input, output=cnn_out)
            print(self.model.summary())

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        # img = image.load_img(image_path, target_size=(299, 299))
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        # x = numpy.array(x)
        # x = numpy.reshape(x, (x.shape[1], x.shape[2], x.shape[3]))
        # y = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        # cv2.imshow("image", y)
        # cv2.waitKey(0)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
