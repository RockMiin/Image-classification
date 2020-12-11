from keras import optimizers
from keras.datasets import cifar10
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16

# set image size & epoch, classes
img_width= 32
img_height= 32

nb_epoch = 30
nb_classes = 10

def vgg16_model(img_width, img_height, nb_epoch, nb_classes):
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(img_width, img_height, 3))

    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, nb_classes)

    y_test = np_utils.to_categorical(y_test, nb_classes)

    # Extract the last layer from third block of vgg16 model
    last = base_model.get_layer('block5_pool').output

    # Add classification layers on top of it
    x = Flatten()(last)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, output)
    # model.summary()

    # model compile & fit
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=nb_epoch,
              batch_size=100,
              verbose=1
              )
    return model

# making model
model= vgg16_model(img_width, img_height, nb_epoch, nb_classes)

# model save
model.save('vgg_model_30.h5')