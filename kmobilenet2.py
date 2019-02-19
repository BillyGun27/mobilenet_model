from keras.layers import Dense, Input, Dropout
from keras.applications.mobilenet_v2 import MobileNetV2


model =MobileNetV2(
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        include_top=True, 
        classes=1000)

model.summary()