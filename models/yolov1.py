import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, DepthwiseConv2D, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Dropout

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
        x = LeakyReLU(alpha=0.1)(x)
        return x

img_size = 448
cells_per_direction = 7 # How many cells
classes = 20
boxes_per_cell = 2
outputs_per_cell = boxes_per_cell * 5 + classes # Depth of final output tensor 
cell_size = img_size / cells_per_direction

model_architecture=[]
model_architecture.append(((cells_per_direction, cells_per_direction),64,2)) # Kernel size, filters, stride 
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
model_architecture.append(("DENSE", cells_per_direction * cells_per_direction * outputs_per_cell))

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
            if layer[1] != cells_per_direction * cells_per_direction * outputs_per_cell:
                x = Dropout(0.1)(x)
                x = LeakyReLU(alpha=0.1)(x)
        else:
            kernel_size, filters, strides = layer[0], layer[1], layer[2]
            x = DepthwiseSeparableConv2D(kernel_size=kernel_size, filters=filters, strides=strides, padding="SAME").call(x)
    output = x
    model_keras = tf.keras.Model(inputs=inputs_to_model, outputs=output, name="yolov1")
    return model_keras


def plot_output(images, output, name):
    import matplotlib.patches as patches

    for idx in range(output.shape[0]):
        single_output = output[idx]
        single_output = np.reshape(single_output, (cells_per_direction, cells_per_direction, outputs_per_cell))
        img = images[idx]
        rectangles = []
        for cell_y in range(cells_per_direction):
            for cell_x in range(cells_per_direction):
                if single_output[cell_y][cell_x][20] > 0.4:
                    x_coord, y_coord, w, h = single_output[cell_y][cell_x][21:25]        

                    # Actual x and y scaled to box size
                    x_coord, y_coord = x_coord*cell_size, y_coord*cell_size
                    # Get actual x and y positions on image
                    x_on_image = cell_size*cell_x + x_coord
                    y_on_image = cell_size*cell_y + y_coord
                    # Bottom left coordinates of the image
                    width_scaled, height_scaled = w * cell_size, h * cell_size
                    x_left = x_on_image - width_scaled / 2
                    y_top = y_on_image - height_scaled / 2
                    rectangles.append(patches.Rectangle((x_left, y_top), width_scaled, height_scaled, linewidth=1, edgecolor='r', facecolor='none'))

                if single_output[cell_y][cell_x][25] > 0.4:
                    x_coord, y_coord, w, h = single_output[cell_y][cell_x][26:30]
                    # Actual x and y scaled to box size
                    x_coord, y_coord = x_coord*cell_size, y_coord*cell_size
                    # Get actual x and y positions on image
                    x_on_image = cell_size*cell_x + x_coord
                    y_on_image = cell_size*cell_y + y_coord
                    # Top left coordinates of the image
                    width_scaled, height_scaled = w * cell_size, h * cell_size
                    x_left = x_on_image - width_scaled / 2
                    y_top = y_on_image - height_scaled / 2
            
                    rectangles.append(patches.Rectangle((x_left, y_top), width_scaled, height_scaled, linewidth=1, edgecolor='r', facecolor='none'))

        # Don't plot if no predictions with high enough probability
        if not rectangles:
            continue
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.imshow(img)
        for rect in rectangles:
            ax.add_patch(rect)
        plt.savefig(f"{name}_model_outputs_{idx}.png")
        plt.clf()


if __name__ == "__main__":
    model = create_model(test_input.shape[1:])
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=100, expand_nested=True, to_file="yolov1.pdf")
    model.summary()

