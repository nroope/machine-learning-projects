import tensorflow as tf
from tensorflow.keras.layers import Flatten



class YoloV1Loss(tf.keras.losses.Loss):


    def __init__(self, cells, boxes, classes, coordinate_loss_weight, no_object_loss_weight):
        self.cells = cells
        self.boxes = boxes
        self.classes = classes
        self.coordinate_loss_weight = coordinate_loss_weight
        self.no_object_loss_weight = no_object_loss_weight

        # Used for logging
        self.height_width_loss = None
        self.center_xy_loss = None
        self.object_loss = None
        self.no_object_loss1 = None
        self.no_object_loss2 = None
        self.class_loss = None
        super().__init__(name="yolov1_loss")

    def call(self, y_true, y_pred):
        return self.losses(y_true, y_pred)


    def intersection_over_union(self, prediction_coordinates, label_coordinates):
        # predictions and label arrays = X, Y, W, H
        box_left_x = prediction_coordinates[:,:,:,0:1] - prediction_coordinates[:,:,:,2:3] / 2
        box_right_x = prediction_coordinates[:,:,:,0:1] - prediction_coordinates[:,:,:,2:3] / 2
        box_top_y = prediction_coordinates[:,:,:,1:2] - prediction_coordinates[:,:,:,3:4] / 2
        box_bottom_y = prediction_coordinates[:,:,:,1:2] - prediction_coordinates[:,:,:,3:4] / 2

        label_left_x = label_coordinates[:,:,:,0:1] - label_coordinates[:,:,:,2:3] / 2
        label_right_x = label_coordinates[:,:,:,0:1] - label_coordinates[:,:,:,2:3] / 2
        label_top_y = label_coordinates[:,:,:,1:2] - label_coordinates[:,:,:,3:4] / 2
        label_bottom_y = label_coordinates[:,:,:,1:2] - label_coordinates[:,:,:,3:4] / 2

        intersection_width = tf.clip_by_value(tf.math.minimum(box_right_x, label_right_x) - tf.math.maximum(box_left_x, label_left_x), 0, tf.float32.max)
        intersection_height = tf.clip_by_value(tf.math.minimum(box_bottom_y, label_bottom_y) - tf.math.maximum(box_top_y, label_top_y), 0, tf.float32.max)
        intersection_area = intersection_width * intersection_height

        box_area = tf.math.abs((box_right_x - box_left_x) * (box_bottom_y - box_top_y))
        label_area = tf.math.abs((label_right_x - label_left_x) * (label_bottom_y - label_top_y))
        union_area = box_area + label_area - intersection_area
        return intersection_area / (union_area + 1e-6)


    def losses(self, labels, predictions):
        """
        Calculate loss. Code based on
        https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py#L34 but done with
        TensorFlow.
        """

        predictions = tf.reshape(predictions, (predictions.shape[0], self.cells, self.cells, self.boxes*5 + self.classes))
        ##### Coordinate losses #####
        loss_fn = tf.keras.losses.MeanSquaredError()
        intersection_over_union_box1 = self.intersection_over_union(predictions[:,:,:, 21:25], labels[:,:,:, 21:25])
        intersection_over_union_box2 = self.intersection_over_union(predictions[:,:,:, 26:30], labels[:,:,:, 21:25])
        intersection_over_union_box1 = tf.expand_dims(intersection_over_union_box1, axis=0)
        intersection_over_union_box2 = tf.expand_dims(intersection_over_union_box2, axis=0)
        intersections = tf.concat([intersection_over_union_box1, intersection_over_union_box2], axis=0)

        boxes_with_higher_iou = tf.cast(tf.math.argmax(intersections, axis=0), intersections.dtype)
        box_exists = labels[:,:,:,20:21]
        box_predictions = box_exists * (boxes_with_higher_iou * predictions[:,:,:, 26:30] + (1 - boxes_with_higher_iou) * predictions[:,:,:, 21:25])
        box_targets = box_exists * labels[:,:,:,21:25]
        box_predictions_width_height_sqrt = tf.math.sign(box_predictions[:,:,:,2:4]) * tf.math.sqrt(tf.math.abs(box_predictions[:,:,:,2:4]) + 1e-6)
        box_targets_width_height_sqrt = tf.math.sqrt(box_targets[:,:,:,2:4])
        
        loss = 0
        self.height_width_loss = self.coordinate_loss_weight*loss_fn(box_predictions_width_height_sqrt, box_targets_width_height_sqrt) # Height/width square root loss
        loss += self.height_width_loss
        self.center_xy_loss = self.coordinate_loss_weight*loss_fn(box_predictions[:,:,:,:2], box_targets[:,:,:,:2]) # Center X/Y loss
        loss += self.center_xy_loss
        ##### Object loss #####
        pred_box = boxes_with_higher_iou * predictions[:,:,:, 25:26] + (1 - boxes_with_higher_iou) * predictions[:,:,:, 20:21]
        self.object_loss = loss_fn(Flatten()(box_exists * pred_box), Flatten()(box_exists * labels[:,:,:, 20:21]))
        loss += self.object_loss
        ##### No object loss #####
        box_exists_flattened = tf.reshape(box_exists, (box_exists.shape[0], box_exists.shape[1] * box_exists.shape[2]))
        preds_flattened_box1 = tf.reshape(predictions[:,:,:, 20:21], (predictions[:,:,:, 20:21].shape[0], predictions[:,:,:, 20:21].shape[1] * predictions[:,:,:, 20:21].shape[2]))
        preds_flattened_box2 = tf.reshape(predictions[:,:,:, 25:26], (predictions[:,:,:, 25:26].shape[0], predictions[:,:,:, 25:26].shape[1] * predictions[:,:,:, 25:26].shape[2]))

        labels_flattened = tf.reshape(labels[:,:,:, 20:21], (labels[:,:,:, 20:21].shape[0], labels[:,:,:, 20:21].shape[1] * labels[:,:,:, 20:21].shape[2]))
        self.no_object_loss1 = self.no_object_loss_weight*loss_fn((1 - box_exists_flattened) * preds_flattened_box1, (1 - box_exists_flattened) * labels_flattened)
        self.no_object_loss2 = self.no_object_loss_weight*loss_fn((1 - box_exists_flattened) * preds_flattened_box2, (1 - box_exists_flattened) * labels_flattened)
        loss += self.no_object_loss1
        loss += self.no_object_loss2


        # Class loss
        self.class_loss = loss_fn(Flatten()(box_exists*predictions[:,:,:,:20]), Flatten()(box_exists*labels[:,:,:,:20]))
        loss += self.class_loss

        return loss

