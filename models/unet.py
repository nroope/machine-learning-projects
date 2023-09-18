import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU, Input

# Code mostly from https://github.com/GoogleCloudPlatform/practical-ml-vision-book/blob/master/04_detect_segment/04b_unet_segmentation.ipynb, just refactored.

class UpsamplingLayer(tf.keras.layers.Layer):

    def __init__(self, filters, size, name, **kwargs):
        self.filters = filters
        self.size = size
        self.strides=2
        super(UpsamplingLayer, self).__init__(name=name, **kwargs)

    def call(self, x):
        x = Conv2DTranspose(self.filters, self.size, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x


def create_unet_model(output_channels):
    input_shape = [128,128,3]
    inputs = Input(shape=input_shape, name='input_image')
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    #               64x64                   32x32                   16x16                   8x8                     4x4
    layer_names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    downsampling_model = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs, name='pretrained_mobilenet')
    downsampling_model.trainable = False

    # Downsample from 128x128 to 4x4
    downsampling_model_outputs = downsampling_model(inputs)
    downsampling_final_output = downsampling_model_outputs[-1]

    # Upsample back to 128x128 
    output = UpsamplingLayer(256, 3, 'upsample_4x4_to_8x8')(downsampling_final_output)
    output = tf.concat([output, downsampling_model_outputs[3]], axis=-1)

    output = UpsamplingLayer(128, 3, 'upsample_8x8_to_16x16')(output)
    output = tf.concat([output, downsampling_model_outputs[2]], axis=-1)

    output = UpsamplingLayer(64, 3, 'upsample_16x16_to_32x32')(output)
    output = tf.concat([output, downsampling_model_outputs[1]], axis=-1)

    output = UpsamplingLayer(32, 3,  'upsample_32x32_to_64x64')(output)
    output = tf.concat([output, downsampling_model_outputs[0]], axis=-1)

    output = Conv2DTranspose(output_channels, 3, strides=2, padding='same', kernel_regularizer="l1")(output) #64x64 -> 128x128
    return tf.keras.Model(inputs=inputs, outputs=output)


def main():
     output_channels = 3
     unet_model = create_unet_model(output_channels=output_channels)
     tf.keras.utils.plot_model(unet_model, show_shapes=True, dpi=100, expand_nested=True, to_file="unet.pdf")

if __name__ == '__main__':
    main()











