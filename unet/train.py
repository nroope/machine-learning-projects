import numpy as np
import tensorflow as tf
from unet import create_unet_model, plot_output, MlFlowLogCallback
from unet_dataloader import get_datasets
import os
import pandas as pd
import argparse
from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size used in training and validation')
    parser.add_argument('--input_shape', default=(448,448,3), nargs='+', type=int, help='Shape of the images used during training')
    parser.add_argument('--data_path', default="", type=str, help='Path to data directory')
    parser.add_argument('--model_type', default="yolov1", type=str, help='Type of the model. Currently only yolov1 supported')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate used')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--no_object_loss_weight', default=0.5, type=float, help='Weight for the no-object loss calculation')
    parser.add_argument('--coordinate_loss_weight', default=5.0, type=float, help='Weight for the coordinate loss')

    return parser



def train(params):
    train_data, test_data, info = get_datasets(params.batch_size)

    train_length = info.splits['train'].num_examples
    steps_per_epoch = train_length // params.batch_size

    model = create_unet_model(output_channels=3)

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    from datetime import datetime
    start = datetime.now()
    print("START", start)
    model.fit(train_data, epochs=params.epochs,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=20,
                          validation_data=test_data, 
                          callbacks=[MlFlowLogCallback()])
    end = datetime.now()
    print("END", end)
    print("Time taken", end-start)
    plot_output(model, test_data, params.batch_size)




def main(params):
    with mlflow.start_run():
        log_params(params.__dict__)
        train(params)



if __name__ == "__main__":
    parser = get_arg_parser()
    params = parser.parse_args()
    main(params)
