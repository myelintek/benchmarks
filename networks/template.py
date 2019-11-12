import model_lib as model


# Entrypoint for framework
class UserModel(model.CNNModel):
  # initial variables including
  # 1. model name
  # 2. default image size (fixel)
  # 3. default batch size
  # 4. default learning rate
  def __init__(self):
    super(UserModel, self).__init__('Template', model.MODEL_TYPE_IMAGE_CLASSIFICATION, 28, 32, 0.005)

  # Where model defined
  def add_inference(self, cnn):
    pass
    # Note: This matches TF's MNIST tutorial model
    #cnn.conv(32, 5, 5)
    #cnn.mpool(2, 2)
    #cnn.conv(64, 5, 5)
    #cnn.mpool(2, 2)
    #cnn.reshape([-1, 64 * 7 * 7])
    #cnn.affine(512)
