"""Lenet model configuration.

References:
  LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner
  Gradient-based learning applied to document recognition
  Proceedings of the IEEE (1998)
"""
import model_lib as model


class UserModel(model.CNNModel):
  def __init__(self, params=None):
    super(UserModel, self).__init__('Lenet5', model.MODEL_TYPE_IMAGE_CLASSIFICATION, 28, 32, 0.005, params=params)

  def add_inference(self, cnn):
    # Note: This matches TF's MNIST tutorial model
    cnn.conv(32, 5, 5)
    cnn.mpool(2, 2)
    cnn.conv(64, 5, 5)
    cnn.mpool(2, 2)
    cnn.reshape([-1, 64 * 7 * 7])
    cnn.affine(512)
