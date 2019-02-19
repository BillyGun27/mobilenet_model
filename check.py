from keras.layers import Input, Lambda
from mobilenet import MobileNet

shape=(None, None, 3)
image_input = shape
model = MobileNet(image_input,20,1.0,True,None)
model.summary()