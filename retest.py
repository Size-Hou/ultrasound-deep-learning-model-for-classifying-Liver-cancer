import pandas as pd
from Model.resnet18 import ResNet
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
from LoadData.LoadData import InputTrainImg, InputTestImg
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm



num_task = 3
width = 128
model = ResNet([2, 2, 2, 2], num_task)
model.build(input_shape = (None, width, width, 3))
optimizer = optimizers.Adam(learning_rate=1e-8)  # 学习率

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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
model.load_weights('Liverre_949_0.75')
model2.load_weights('Liver2_100_0.95')
model4.load_weights('Liver4_69_0.60')
model7.load_weights('Liver7_1_0.39')

dat = InputTestImg('D:/projects/Liver/test', width)
test_x, test_y = dat.load_test_data()

score = model.evaluate(test_x, test_y)
score = model2.evaluate(test_x, test_y)
score = model4.evaluate(test_x, test_y)
score = model7.evaluate(test_x, test_y)
pre_y = np.argmax(model.predict(test_x), axis=1)
pre_y2 = np.argmax(model2.predict(test_x), axis=1)
pre_y4 = np.argmax(model4.predict(test_x), axis=1)
pre_y7 = np.argmax(model7.predict(test_x), axis=1)

c = confusion_matrix(test_y, pre_y)
c2 = confusion_matrix(test_y, pre_y2)
c4 = confusion_matrix(test_y, pre_y4)
c7 = confusion_matrix(test_y, pre_y7)
print(c)
print(c2)
print(c4)
print(c7)
# plt.figure()
# classes = ['CHC', 'HCC', 'ICC']
#
# plt.imshow(c, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
# plt.title('Confusion Matrix of Liver Tumour')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=-45)
# plt.yticks(tick_marks, classes)
#
# thresh = c.max() / 2.
# iters = np.reshape([[[i, j] for j in range(3)] for i in range(3)], (c.size, 2))
# for i, j in iters:
#     plt.text(j, i, format(c[i, j]))  # 显示对应的数字
#
# plt.ylabel('Real label')
# plt.xlabel('Prediction')
# plt.tight_layout()
# plt.savefig('CHCHCCICC.svg', format='svg')
# plt.show()

print(precision_score(test_y, pre_y, average='macro'))
print(recall_score(test_y, pre_y, average='macro'))
print(f1_score(test_y, pre_y, average='macro'))
print(precision_score(test_y, pre_y2, average='macro'))
print(recall_score(test_y, pre_y2, average='macro'))
print(f1_score(test_y, pre_y2, average='macro'))
print(precision_score(test_y, pre_y4, average='macro'))
print(recall_score(test_y, pre_y4, average='macro'))
print(f1_score(test_y, pre_y4, average='macro'))
print(precision_score(test_y, pre_y7, average='macro'))
print(recall_score(test_y, pre_y7, average='macro'))
print(f1_score(test_y, pre_y7, average='macro'))
# print(test_y)
# print(pre_y)
print(roc_auc_score(test_y, model.predict(test_x), multi_class='ovo'))
print(roc_auc_score(test_y, model.predict(test_x), multi_class='ovr'))
print(roc_auc_score(test_y, model2.predict(test_x), multi_class='ovo'))
print(roc_auc_score(test_y, model2.predict(test_x), multi_class='ovr'))
print(roc_auc_score(test_y, model4.predict(test_x), multi_class='ovo'))
print(roc_auc_score(test_y, model4.predict(test_x), multi_class='ovr'))
print(roc_auc_score(test_y, model7.predict(test_x), multi_class='ovo'))
print(roc_auc_score(test_y, model7.predict(test_x), multi_class='ovr'))
# print(train_x.shape)
# print(final_pre)
# print(test_y)
# print(test_z[test_y == 0])
# print("test_score: ", score)
# print(test_z[final_pre != test_y])
# print(test_y[final_pre != test_y])
# print(test_z[final_pre != test_y].shape)
# First aggregate all false positive rates

y = label_binarize(test_y, classes=[0, 1, 2])
n_classes = y.shape[1]
pred_y = model.predict(test_x)
pred_y2 = model2.predict(test_x)
pred_y4 = model4.predict(test_x)
pred_y7 = model7.predict(test_x)
# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()
fpr4 = dict()
tpr4 = dict()
roc_auc4 = dict()
fpr7 = dict()
tpr7 = dict()
roc_auc7 = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], pred_y[:, i])
    fpr2[i], tpr2[i], _ = roc_curve(y[:, i], pred_y2[:, i])
    fpr4[i], tpr4[i], _ = roc_curve(y[:, i], pred_y4[:, i])
    fpr7[i], tpr7[i], _ = roc_curve(y[:, i], pred_y7[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
    roc_auc4[i] = auc(fpr4[i], tpr4[i])
    roc_auc7[i] = auc(fpr7[i], tpr7[i])

# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), pred_y.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in range(n_classes)]))
all_fpr4 = np.unique(np.concatenate([fpr4[i] for i in range(n_classes)]))
all_fpr7 = np.unique(np.concatenate([fpr7[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
mean_tpr2 = np.zeros_like(all_fpr2)
mean_tpr4 = np.zeros_like(all_fpr4)
mean_tpr7 = np.zeros_like(all_fpr7)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr2 += interp(all_fpr2, fpr2[i], tpr2[i])
    mean_tpr4 += interp(all_fpr4, fpr4[i], tpr4[i])
    mean_tpr7 += interp(all_fpr7, fpr7[i], tpr7[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
mean_tpr2 /= n_classes
mean_tpr4 /= n_classes
mean_tpr7 /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
fpr2["macro"] = all_fpr2
tpr2["macro"] = mean_tpr2
roc_auc2["macro"] = auc(fpr2["macro"], tpr2["macro"])
fpr4["macro"] = all_fpr4
tpr4["macro"] = mean_tpr4
roc_auc4["macro"] = auc(fpr4["macro"], tpr4["macro"])
fpr7["macro"] = all_fpr7
tpr7["macro"] = mean_tpr7
roc_auc7["macro"] = auc(fpr7["macro"], tpr7["macro"])

# Plot all ROC curves
# plt.figure()
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="ROC curve of Our Method(area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle="-",
    linewidth=3,
)
plt.plot(
    fpr2["macro"],
    tpr2["macro"],
    label="ROC curve of MobileNet(area = {0:0.2f})".format(roc_auc2["macro"]),
    color="aqua",
    linestyle="-",
    linewidth=3,
)
plt.plot(
    fpr4["macro"],
    tpr4["macro"],
    label="ROC curve of DenseNet121(area = {0:0.2f})".format(roc_auc4["macro"]),
    color="darkorange",
    linestyle="-",
    linewidth=3,
)
plt.plot(
    fpr7["macro"],
    tpr7["macro"],
    label="ROC curve of InceptionV3(area = {0:0.2f})".format(roc_auc7["macro"]),
    color="deeppink",
    linestyle="-",
    linewidth=3,
)
lw = 2
# colors = cycle(["aqua", "darkorange", "cornflowerblue"])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=lw,
#         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#     )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC of Liver Cancer")
plt.legend(loc="lower right")
plt.savefig('ROCfinal.svg',format='svg')
plt.show()

def bootstrap_auc(y, pred, classes, bootstraps = 10000, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # print(df)
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

pre_yy = label_binarize(pre_y, classes=[0, 1, 2])
pre_yy2 = label_binarize(pre_y2, classes=[0, 1, 2])
pre_yy4 = label_binarize(pre_y4, classes=[0, 1, 2])
pre_yy7 = label_binarize(pre_y7, classes=[0, 1, 2])
# statistics = bootstrap_auc(y, pre_yy, ['CHC', 'HCC', 'ICC'])
# print(statistics.shape)
# print(statistics)
# print(statistics.mean(axis=1))
# print(statistics.mean(axis=1)+statistics.std(axis=1))
# print(statistics.mean(axis=1)-statistics.std(axis=1))

print(np.argmax(pred_y,axis=1))
print(np.argmax(pred_y2,axis=1))
print(np.argmax(pred_y4,axis=1))
print(np.argmax(pred_y7,axis=1))
pd.DataFrame(pred_y).to_csv('p1.csv')
pd.DataFrame(pred_y2).to_csv('p2.csv')
pd.DataFrame(pred_y4).to_csv('p4.csv')
pd.DataFrame(pred_y7).to_csv('p7.csv')
pd.DataFrame(y).to_csv('y.csv')
print(y)



# img_path = 'D:/projects/Liver/train/HCC/SHI HONGMEICEUS LIVER20171120153627591.JPG'

model_builder = model2
img_size = (128, 128)
last_conv_layer_name = "conv_pw_5"
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    box = (0, 50, 950, 850)
    img = img.crop(box)
    img = img.resize((128, 128))
    img = img.convert('RGB')
#     img = image.img_to_array(img,dtype='float32')
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # 首先，我们创建一个模型，将输入图像映射到最后一个conv层的激活以及输出预测
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    #然后，我们为输入图像计算top预测类关于最后一个conv层的激活的梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        #如果没有传入pred_index,就计算pred[0]中最大的值对应的下标号index
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)这是输出神经元(预测概率最高的或者选定的那个)对最后一个卷积层输出特征图的梯度
    # with regard to the output feature map of the last conv layer
    # grads.shape(1, 10, 10, 2048)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient这是一个向量,每一项都是 指定特征图通道上的平均值
    # over a specific feature map channel
    # pooled_grads 是一个一维向量,shape=(2048,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    # last_conv_layer_output[0]是一个三维的卷积层 ,@矩阵相乘(点积)
    #last_conv_layer_output.shape  =(10, 10, 2048)
    last_conv_layer_output = last_conv_layer_output[0]
    #heatmap (10, 10, 1) = (10, 10, 2048)  @(2048,)相当于(10, 10, 2048)乘以(2048,1)
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # tf.squeeze 去除1的维度,(10, 10)
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # tf.maximum(heatmap, 0) 和0比较大小,返回一个>=0的值,相当于relu,然后除以heatmap中最大的 值,进行normalize归一化到0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

model2.summary()

img_array = get_img_array('D:/projects/Liver/train/ICC/20180510083645785.JPG', size=(224,224))
model2.layers[-1].activation = None

# Print what the top predicted class is
preds = model2.predict(img_array)
# print("Predicted:", decode_predictions(preds, top=1)[0])#这些地方所加的0皆是batch中第一个sample的意思
last_conv_layer_name = "conv_pw_13_relu"
# Generate class activation heatmapconv2_block3_outconv3_block8_out
heatmap = make_gradcam_heatmap(img_array, model2, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)
# plt.colorbar()
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉纵坐标值
# plt.show()
# plt.colorbar()
plt.savefig('CHC11.svg')

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    box = (0, 50, 950, 850)
    img = img.crop(box)
    img = img.resize((128, 128))
    img = img.convert('RGB')
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose重叠 the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
#     display(Image(cam_path))
    a = cam_path
    return a

a = save_and_display_gradcam('D:/projects/Liver/train/ICC/20180510083645785.JPG', heatmap)
# display(Image(a))
a = plt.imread(a)
# plt.colorbar()
plt.imshow(a)
plt.colorbar()
# plt.xticks([])  #去掉横坐标值
# plt.yticks([])  #去掉纵坐标值
plt.title('ICC')
plt.savefig('ICC.svg')
