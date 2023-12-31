import tensorflow as tf
from yolov1_loss import YoloV1Loss
from yolov1 import create_model, plot_output
from yolov1_dataloader import YOLOV1DataLoader
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



def train(model, train_data, test_data, params):
    loss_fn = YoloV1Loss(coordinate_loss_weight=params.coordinate_loss_weight, no_object_loss_weight=params.no_object_loss_weight)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    epochs = params.epochs
    steps_per_epoch = len(train_data)
    test_data_iterator = iter(test_data)

    @tf.function
    def training_step(x_train, y_train):
        with tf.GradientTape() as tape:
            model_output = model(x_train, training=True)
            loss_value = loss_fn(y_train, model_output)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value


    @tf.function
    def validation_step(x_test, y_test):
        model_output = model(x_test, training=False)
        loss_value = loss_fn(y_test, model_output)
        return loss_value

    from datetime import datetime
    start = datetime.now()
    print("START", start)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (x_train, y_train) in enumerate(train_data):
            log_step = epoch*steps_per_epoch + step
            loss_value = training_step(x_train, y_train)

            if log_step % 10 == 0:
                log_metric("Training loss", loss_value, step=log_step)
            if log_step % 50 == 0:
                # Validate every 50 batches
                x_test, y_test = next(test_data_iterator)
                loss_value = validation_step(x_test, y_test)
                log_metric("Validation loss", loss_value, step=log_step)

        train_data.shuffle()
    end = datetime.now()
    print("END", end)
    print("Time taken", end-start)
    x_test, y_test = next(test_data_iterator)
    model_output = model(x_test, training=False)
    model_output = model_output.numpy()
    x_test = x_test.numpy()

    plot_output(x_test, model_output, params.model_type)




def main(params):
    train_df = pd.read_csv(os.path.join(params.data_path, "train.csv"), names=["image", "label_file"])
    train_data = YOLOV1DataLoader(train_df, params.batch_size, params.input_shape, params.data_path)

    test_df = pd.read_csv(os.path.join(params.data_path, "test.csv"), names=["image", "label_file"])
    test_data = YOLOV1DataLoader(test_df, params.batch_size, params.input_shape, params.data_path)
    model = create_model(input_shape=params.input_shape)

    with mlflow.start_run():
        log_params(params.__dict__)
        train(model, train_data, test_data, params)



if __name__ == "__main__":
    parser = get_arg_parser()
    params = parser.parse_args()
    main(params)
