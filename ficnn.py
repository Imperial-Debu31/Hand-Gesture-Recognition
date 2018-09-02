import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(.25))


# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(.20))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(.25))

classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(.5))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_set = train_datagen.flow_from_directory(
    'dataset/training',
    target_size=(48, 48),
    batch_size=16,
    color_mode='grayscale',
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/testing',
    target_size=(48, 48),
    batch_size=16,
    class_mode='binary',
    color_mode='grayscale')

classifier.fit_generator(
    train_set,
    steps_per_epoch=8575,
    epochs=5,
    validation_data=test_set,
    validation_steps=2000)
print('done!!')

from keras.models import load_model

classifier = load_model('cdi.h5')

import numpy as np
from keras.preprocessing import image
import cv2

img = cv2.imread('spread//8.jpg',0)
img2 =cv2.imread('spread//8.jpg')
imgbw = cv2.resize(img, (48, 48))
imgbw = image.img_to_array(imgbw)
imgbw = np.expand_dims(imgbw, axis=0)
result = classifier.predict(imgbw)
# print(result)

if result[0][0] == 0:
    print('0')
    cv2.putText(img2, 'Symbol: One', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)

else:
    print('1')
    cv2.putText(img2, 'Symbol: Victory', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)

cv2.imshow('Output',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''classifier.save('cdi.h5')
classifier.save_weights('cdiw.h5')
jstring = classifier.to_json()
jfile = open('classifier.json','w')
jfile.write(jstring)
jfile.close()'''
