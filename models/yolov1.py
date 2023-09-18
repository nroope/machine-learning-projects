import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, DepthwiseConv2D, Conv2D, MaxPooling2D, Flatten, Dense

class DepthwiseSeparableConv2D():

    def __init__(self, filters, kernel_size, strides, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides=strides
        super(DepthwiseSeparableConv2D, self).__init__()

    def call(self, x):
        x = DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = Conv2D(filters=self.filters, kernel_size=(1,1), strides=self.strides, padding="SAME")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x


S = 7 # How many cells
C = 20 # How many classes
boxes_per_cell = 2
D = boxes_per_cell * 5 + C # Depth of final output tensor 

test_input = np.random.normal(size=(6,448,448,3))*256
test_input = np.reshape(test_input, (6,448,448,3))
model_architecture=[]
model_architecture.append(((S,S),64,2)) # Kernel size, filters, stride 
model_architecture.append("MAXPOOL")
model_architecture.append(((3,3),192,1))
model_architecture.append("MAXPOOL")
model_architecture.append(((1,1),128,1))
model_architecture.append(((3, 3),256,1))
model_architecture.append(((1,1),256,1))
model_architecture.append(((3,3),512,1))
model_architecture.append("MAXPOOL")
model_architecture.append(((1,1),256,1))
model_architecture.append(((3,3),512,1))
model_architecture.append(((1,1),256,1)) 
model_architecture.append(((3,3),512,1))
model_architecture.append(((1,1),256,1))
model_architecture.append(((3,3),512,1)) 
model_architecture.append(((1,1),256,1)) 
model_architecture.append(((3,3),512,1)) 
model_architecture.append(((1,1),512,1)) 
model_architecture.append(((3,3),1024,1))
model_architecture.append("MAXPOOL")
model_architecture.append(((1,1),512,1))
model_architecture.append(((3,3),1024,1))
model_architecture.append(((1,1),512,1))
model_architecture.append(((3,3),1024,1))
model_architecture.append(((1,1),512,1))
model_architecture.append(((3,3),1024,1))
model_architecture.append(((1,1),512,1))
model_architecture.append(((3,3),1024,1))
model_architecture.append(((3,3),1024,1)) 
model_architecture.append(((3,3),1024,2))
model_architecture.append(((3,3),1024,1))
model_architecture.append(((3,3),1024,1))
model_architecture.append(("DENSE", 4096))
model_architecture.append(("DENSE", S*S*D))

def create_model(input_shape):
    inputs_to_model = tf.keras.Input(shape=input_shape,name="input_data")
    x = inputs_to_model
    for layer in model_architecture:
        if "MAXPOOL" in layer:
            x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        elif "DENSE" in layer[0]:
            # Flatten if necessary
            if len(x.shape) > 2:
                x = Flatten()(x)
            x = Dense(layer[1])(x)
        else:
            kernel_size, filters, strides = layer[0], layer[1], layer[2]
            x = DepthwiseSeparableConv2D(kernel_size=kernel_size, filters=filters, strides=strides, padding="SAME").call(x)
    output = x
    model_keras = tf.keras.Model(inputs=inputs_to_model, outputs=output, name="yolov1")
    return model_keras


if __name__ == "__main__":
    model = create_model(test_input.shape[1:])
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=100, expand_nested=True, to_file="yolov1.pdf")
    model.summary()

