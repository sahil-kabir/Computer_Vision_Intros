from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        # reshape tensors, reduce magnitude of elements in tensors
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.data = History()

        network = models.Sequential() # construction of neural network
        network.add(layers.Conv2D(128, (3,3), activation = 'relu', input_shape = (32, 32, 3))) # input layer, convolution for feature extraction
        network.add(layers.MaxPool2D((2,2))) # max pool to filter/enhance feature extractions
        network.add(layers.Conv2D(128, (3,3), activation = 'relu'))
        network.add(layers.MaxPool2D((2,2)))
        network.add(layers.Conv2D(128, (3,3), activation = 'relu'))
        network.add(layers.MaxPool2D((2,2)))
        network.add(layers.Flatten()) # flatten for tensors to be moved into dense layers
        network.add(layers.Dense(64, activation = 'relu'))
        network.add(layers.Dense(64, activation = 'relu'))
        network.add(layers.Dense(10, activation = 'softmax')) # softmax activation for multiclass single-label classification

        self.network = network

    def fit(self, X_train, y_train, X_test, y_test, train_reshape, test_reshape):
        self.X_train = X_train.reshape(train_reshape) 
        self.X_train = self.X_train.astype('float32')/255
        self.X_test = X_test.reshape(test_reshape)
        self.X_test = self.X_test.astype('float32')/255
        
        
        self.y_train = to_categorical(y_train) 
        self.y_test = to_categorical(y_test)

        # compile the network 
        self.network.compile(optimizer = 'rmsprop', # a standard optimizor
                        loss = 'categorical_crossentropy',  # loss function for multiclass single-label classification
                        metrics = ['accuracy'])
        # Fit the data through the network, 
        self.data = self.network.fit(self.X_train, self.y_train, epochs = 8, batch_size = 75, validation_data = (self.X_test, self.y_test))

    def display_fitting(self):
        # construct pyplot to compare training accuracy and validation accuracy
        plt.plot(range(1, len(self.data.history['accuracy']) + 1), self.data.history['accuracy'], label = 'Training Accuracy')
        plt.plot(range(1, len(self.data.history['val_accuracy']) + 1), self.data.history['val_accuracy'], label = 'Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()



# load data - 'test' items will be used for validation data
(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()
model = Model()
model.fit(trainImages, trainLabels, testImages, testLabels, (50000, 32, 32, 3), (10000, 32, 32, 3))
model.display_fitting()
