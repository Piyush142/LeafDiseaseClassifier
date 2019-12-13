#%%
import numpy as np
import os
import time
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from imagenet_utils import preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#%%
# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

# Define the number of classes
num_classes = 38
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:630]=0
labels[630:1251]=1
labels[1251:1526]=2
labels[1526:3171]=3
labels[3171:4673]=4
labels[4673:5527]=5
labels[5527:6579]=6
labels[6579:7092]=7
labels[7092:8284]=8
labels[8284:9446]=9
labels[9446:10431]=10
labels[10431:11611]=11
labels[11611:12994]=12
labels[12994:13417]=13
labels[13417:14493]=14
labels[14493:20000]=15
labels[20000:22297]=16
labels[22297:22657]=17
labels[22657:23654]=18
labels[23654:25132]=19
labels[25132:26132]=20
labels[26132:26284]=21
labels[26284:27284]=22
labels[27284:27655]=23
labels[27655:32745]=24
labels[32745:34580]=25
labels[34580:35036]=26
labels[35036:36226]=27
labels[36226:38353]=28
labels[38353:39353]=29
labels[39353:40944]=30
labels[40944:42853]=31
labels[42853:43805]=32
labels[43805:45576]=33
labels[45576:47252]=34
labels[47252:48656]=35
labels[48656:49029]=36
labels[49029:54386]=37

names = ['Apple__Apple_scab','Apple__Black_rot','Apple__Cedar_apple_rust','Apple__healthy','Blueberry__healthy',
         'Cherry_(including_sour)__healthy','Cherry_(including_sour)__Powdery_mildew','Corn_(maize)__Cercospora_leaf_spotGray_leaf_spot',
         'Corn_(maize)__Northern_Leaf_Blight','Grape__Black_rot','Grape__Esca_(Black_Measles)','Grape__healthy',
         'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)','Orange_Haunglongbing_(Citrus_greening)','Peach__Bacterial_spot'
         ,'Peach__healthy','Pepper,_bell__Bacterial_spot','Pepper,_bell__healthy','Potato__Early_blight','Potato__healthy',
         'Potato__Late_blight','Raspberry__healthy','Soybean__healthy','Squash__Powdery_mildew','Strawberry__healthy',
         'Strawberry__Leaf_scorch','Tomato__Bacterial_spot','Tomato__Early_blight','Tomato__healthy','Tomato__Late_blight',
         'Tomato__Leaf_Mold','Tomato__Septoria_leaf_spot','Tomato__Spider_mitesTwo-spotted_spider_mite','Tomato__Target_Spot',
         'Tomato__Tomato_mosaic_virus','Tomato__Tomato_Yellow_Leaf_Curl_Virus']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

###########################################################################################################################
#%%
# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
custom_resnet_model.save("Disease.model")
###########################################################################################################################
##%%
## Fine tune the resnet 50
#image_input = Input(shape=(224, 224, 3))
#model = ResNet50(input_tensor=image_input , weights='imagenet',include_top=False)
#model.summary()
#last_layer = model.output
## add a global spatial average pooling layer
#x = GlobalAveragePooling2D()(last_layer)
## add fully-connected & dropout layers
#x = Dense(512, activation='relu',name='fc-1')(x)
#x = Dropout(0.5)(x)
#x = Dense(256, activation='relu',name='fc-2')(x)
#x = Dropout(0.5)(x)
## a softmax layer for 4 classes
#out = Dense(num_classes, activation='softmax',name='output_layer')(x)
#
## this is the model we will train
#custom_resnet_model2 = Model(inputs=model.input, outputs=out)
#
#custom_resnet_model2.summary()
#
#for layer in custom_resnet_model2.layers[:-6]:
#	layer.trainable = False
#
#custom_resnet_model2.layers[-1].trainable
#
#custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#
#t=time.time()
#hist = custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))
#print('Training time: %s' % (t - time.time()))
#(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)
#
#print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
################################################################################
## Adding to save model
#
#custom_resnet_model2.save("Disease_Fine.model")
############################################################################################
