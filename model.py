#Loading library
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# Change to locate path and Read image 
def read_image(source_path):
	filename = source_path.split('/')[-1]
	current_path = '/home/carnd/Training_data1/IMG/' + filename
	image = cv2.imread(current_path)
	return image


# Open csv file
lines = []
with open('/home/carnd/Training_data1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# split dataset to 80% training data, 20% validation data
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

# Generator with coroutine
def generator(samples, batch_size = 32):
	num_samples = len(samples)
	while 1:
		# shuffle data
		sklearn.utils.shuffle(samples)
		#Loop over batches
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			measurements = []
			for batch_sample in batch_samples:
				# read camera images
				image_center = read_image(batch_sample[0])	
				image_left = read_image(batch_sample[1])
				image_right = read_image(batch_sample[2])
				images.append(image_center)
				images.append(image_left)
				images.append(image_right)
				# left and right camera correctness
				correction = 0.1
				steering_center = float(batch_sample[3])
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				measurements.append(steering_center)
				measurements.append(steering_left)
				measurements.append(steering_right)

			
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#Normalize the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#Crop images to 80X320X3
model.add(Cropping2D(cropping=((60,20),(0,0))))
#Nvidia Network
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = 'relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = 'relu'))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation = 'relu'))
model.add(Convolution2D(64,3,3,subsample=(1,1),activation = 'relu'))
model.add(Convolution2D(64,3,3,subsample=(1,1),activation = 'relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

#Optimize MSE using ADAM optimizer
model.compile(loss='mse', optimizer='adam')
train_steps = np.ceil(len(train_samples)/32).astype(np.int32)
validation_steps =np.ceil(len(validation_samples)/32).astype(np.int32)

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch = 5)


model.save('model.h5')
