import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from sklearn.model_selection import KFold
import numpy as np
from LoadData.LoadData import InputTrainImg, InputTestImg
from Model.resnet18 import ResNet, train_model
from tensorflow.keras import layers, Sequential, optimizers
num_task = 3
width = 128
datasets = InputTrainImg('D:/projects/Liver/data', width)
x_train, y_train, val_x, val_y, val_z= datasets.load_train_data()


model = ResNet([2, 2, 2, 2], num_task)
model.build(input_shape = (None, width, width, 3))
optimizer = optimizers.Adam(learning_rate=1e-3)  # 学习率
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.load_weights('Liverre_949_0.75')

base_model2 = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(width,width,3))
base_model2.trainable = False
x = base_model2.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(8,activation='relu')(x)
x = tf.keras.layers.Dense(3,activation="softmax")(x)
model2 = tf.keras.Model(inputs=base_model2.input, outputs=x)
model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

base_model4 = DenseNet121(weights='imagenet', include_top=False, input_shape=(width, width, 3))
base_model4.trainable = False
x = base_model4.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(8,activation='relu')(x)
x = tf.keras.layers.Dense(3,activation="softmax")(x)
model4 = tf.keras.Model(inputs=base_model4.input, outputs=x)


model4.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

base_model7 = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, width, 3))
base_model7.trainable = False
x = base_model7.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(8,activation='relu')(x)
x = tf.keras.layers.Dense(3,activation="softmax")(x)
model7 = tf.keras.Model(inputs=base_model7.input, outputs=x)
model7.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

seed = 6
np.random.seed(seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train, y_train)

trscores = []
trainscores = []
cvscores = []
scores = []
trscores2 = []
trainscores2 = []
cvscores2 = []
scores2 = []
trscores4 = []
trainscores4 = []
cvscores4 = []
scores4 = []
trscores7 = []
trainscores7 = []
cvscores7 = []
scores7 = []
for k, (train, val) in enumerate(kfold):
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='Liver_{epoch}_{accuracy:.2f}',
        save_best_only=True,
        monitor='accuracy',
        save_weights_only=True,
        verbose=1)]
    model.fit(np.array(x_train)[train] ,np.array(y_train)[train], epochs=2, callbacks=callbacks, batch_size=10)
    trainscores = model.evaluate(np.array(x_train)[train], np.array(y_train)[train])
    scores = model.evaluate(np.array(x_train)[val], np.array(y_train)[val])
    # y_p = np.argmax(model.predict(x_test),axis=1)
    # model.evaluate(x_test, y_test)
    # y_pred.append(y_p)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    trscores.append(trainscores[1] * 100)
    callbacks2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='Liver2_{epoch}_{accuracy:.2f}',
        save_best_only=True,
        monitor='accuracy',
        save_weights_only=True,
        verbose=1)]
    model2.fit(np.array(x_train)[train] ,np.array(y_train)[train], epochs=100, callbacks=callbacks2, batch_size=10)
    trainscores2 = model2.evaluate(np.array(x_train)[train], np.array(y_train)[train])
    scores2 = model2.evaluate(np.array(x_train)[val], np.array(y_train)[val])
    # y_p2 = np.argmax(model2.predict(x_test),axis=1)
    # model2.evaluate(x_test, y_test)
    # y_pred2.append(y_p2)
    print("%s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))
    cvscores2.append(scores2[1] * 100)
    trscores2.append(trainscores2[1] * 100)
    callbacks4 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='Liver4_{epoch}_{accuracy:.2f}',
        save_best_only=True,
        monitor='accuracy',
        save_weights_only=True,
        verbose=1)]
    model4.fit(np.array(x_train)[train] ,np.array(y_train)[train], epochs=100, callbacks=callbacks4, batch_size=10)
    trainscores4 = model4.evaluate(np.array(x_train)[train], np.array(y_train)[train])
    scores4 = model4.evaluate(np.array(x_train)[val], np.array(y_train)[val])
    # y_p4 = np.argmax(model4.predict(x_test),axis=1)
    # model4.evaluate(x_test, y_test)
    # y_pred4.append(y_p4)
    print("%s: %.2f%%" % (model4.metrics_names[1], scores4[1]*100))
    cvscores4.append(scores4[1] * 100)
    trscores4.append(trainscores4[1] * 100)
    callbacks7 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='Liver7_{epoch}_{accuracy:.2f}',
        save_best_only=True,
        monitor='accuracy',
        save_weights_only=True,
        verbose=1)]
    model7.fit(np.array(x_train)[train] ,np.array(y_train)[train], epochs=100, callbacks=callbacks7, batch_size=10)
    trainscores7 = model7.evaluate(np.array(x_train)[train], np.array(y_train)[train])
    scores7 = model7.evaluate(np.array(x_train)[val], np.array(y_train)[val])
    # y_p7 = np.argmax(model7.predict(x_test),axis=1)
    # model7.evaluate(x_test, y_test)
    # y_pred7.append(y_p7)
    print("%s: %.2f%%" % (model7.metrics_names[1], scores7[1]*100))
    cvscores7.append(scores7[1] * 100)
    trscores7.append(trainscores7[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(trscores), np.std(trscores)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(trscores2), np.std(trscores2)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(trscores4), np.std(trscores4)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores4), np.std(cvscores4)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(trscores7), np.std(trscores7)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores7), np.std(cvscores7)))