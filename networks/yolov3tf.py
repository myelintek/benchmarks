import model_lib as model
import tensorflow as tf

from tensorflow.layers import (
    conv2d,
    batch_normalization
)

from tensorflow.contrib.layers import l2_regularizer as l2


YOLO_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]


def darknet_conv2d_bn_leaky(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4), 'use_bias': False}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)

    conv1 = conv2d(*args, **darknet_conv_kwargs)
    conv1_bn = batch_normalization(conv1)
    conv1_bn_leaky = tf.leaky_relu(conv1_bn, alpha=0.1)
    return conv1_bn_leaky


def resblock_body(x, num_filters, num_blocks):
    x = tf.pad(x, tf.constant([[1, 0], [1, 0]]), 'CONSTANT')
    x = darknet_conv2d_bn_leaky(x, num_filters, (3, 3), strides=(2, 2))
    for i in range(num_blocks):
        y = darknet_conv2d_bn_leaky(x, num_filters//2, (1, 1))
        y = darknet_conv2d_bn_leaky(y, num_filters, (3, 3))
        x = tf.math.add(x, y)
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = darknet_conv2d_bn_leaky(x, 32, (3, 3))
    x64 = resblock_body(x, 64, 1)
    x128 = resblock_body(x64, 128, 2)
    x256 = resblock_body(x128, 256, 8)
    x512 = resblock_body(x256, 512, 8)
    x1024 = resblock_body(x512, 1024, 4)
    return x256, x512, x1024


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''

    x = darknet_conv2d_bn_leaky(x, num_filters, (1, 1))
    x = darknet_conv2d_bn_leaky(x, num_filters*2, (3, 3))
    x = darknet_conv2d_bn_leaky(x, num_filters, (1, 1))
    x = darknet_conv2d_bn_leaky(x, num_filters*2, (3, 3))
    x = darknet_conv2d_bn_leaky(x, num_filters, (1, 1))

    y = darknet_conv2d_bn_leaky(x, num_filters*2, (3, 3))
    y = darknet_conv2d_bn_leaky(y, out_filters, (1, 1))
    return x, y


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = tf.shape(feats)[1:3]  # height, width
    grid_y = tf.tile(
        tf.reshape(tf.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(
        tf.reshape(tf.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], -1)
    grid = tf.cast(grid, feats.dtype)

    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""

    darknet_256, darknet_512, darknet_1024 = darknet_body(inputs)
    x, y1 = make_last_layers(darknet_1024, 512, num_anchors*(num_classes+5))

    x = darknet_conv2d_bn_leaky(x, 256, (1, 1))
    # TODO chck resize with shape or not ( w:2,h:2 or w:in*2,h: in*2)
    x = tf.images.resize_images(x, 2, 2, None)

    x = tf.concat([x, darknet_512], -1)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = darknet_conv2d_bn_leaky(x, 128, (1, 1))
    # TODO chck resize with shape or not ( w:2,h:2 or w:in*2,h: in*2)
    x = tf.images.resize_images(x, 2, 2, None)

    x = tf.concat([x, darknet_256], -1)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return y1, y2, y3


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    num_layers = len(anchors)//3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, y_true[0].dtype)
    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], y_true[0].dtype) for l in range(num_layers)]
    loss = 0
    m = tf.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = tf.cast(m, yolo_outputs[0].dtype)

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(
                yolo_outputs[l],
                anchors[anchor_mask[l]],
                num_classes,
                input_shape,
                calc_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh], -1)

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = tf.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = tf.case(object_mask, raw_true_wh, tf.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3]*y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = tf.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, true_box.dtype))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * tf.sigmoid_cross_entropy_with_logits(raw_true_xy, raw_pred[..., 0:2])
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh-raw_pred[..., 2:4])
        confidence_loss = object_mask * tf.sigmoid_cross_entropy_with_logits(object_mask, raw_pred[..., 4:5]) + \
            (1-object_mask) * tf.sigmoid_cross_entropy_with_logits(object_mask, raw_pred[..., 4:5]) * ignore_mask
        class_loss = object_mask * tf.sigmoid_cross_entropy_with_logits(true_class_probs, raw_pred[..., 5:])

        xy_loss = tf.sum(xy_loss) / mf
        wh_loss = tf.sum(wh_loss) / mf
        confidence_loss = tf.sum(confidence_loss) / mf
        class_loss = tf.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(
                    loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, tf.sum(ignore_mask)], message='loss: ')
    return loss


class UserModel(model.NNModel):
    def __init__(self, params=None):
        super(UserModel, self).__init__('YoloV3TF', model.MODEL_TYPE_OBJECT_DETECTION, 416, params=params)
        self.num_class = 0

    def add_inference(self, inputs, training, nclass):
        self.num_class = nclass
        y1, y2, y3 = yolo_body(inputs, len(YOLO_ANCHORS), nclass)
        return [y1, y2, y3]

    def add_post_inference(self):
        pass

    def loss_function(self, model_output, lables):
        return yolo_loss(model_output.logits, YOLO_ANCHORS, self.num_class)
