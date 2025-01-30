#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


# In[16]:


def load_data(low_light, high_light):
    images = []
    labels= []
    
    #load low light images
    for filename in os.listdir(low_light):
        if filename.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(low_light, filename)
            image = cv2.imread(img_path)
            if image is not None:
                #resize the image to a standard size (e.g., 64 X 64)
                image = cv2.resize(image, (64,64))
                images.append(image)
                labels.append(0) #label for low light
    #load hig light images
    for filename in os.listdir(high_light):
        if filename.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(high_light, filename)
            image = cv2.imread(img_path)
            if image is not None:
                #resize the image to a standard size (e.g., 64x64)
                image = cv2.resize(image, (64,64))
                images.append(image)
                labels.append(1) #labels for high light
                
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# In[21]:


#specify paths to the folder
low_light = r"D:\Notes\Mtech- Ai and Ml\INT522 Python For Machine Learning\project\low_light"
high_light = r"D:\Notes\Mtech- Ai and Ml\INT522 Python For Machine Learning\project\high_light"
images, labels = load_data(low_light, high_light)


# In[22]:


#normalize the images
images = images / 255.0 #scale pixel values to [0,1]


# In[23]:


#convert labels to categorical (binary classification)
labels = to_categorical(labels, num_classes=2)


# In[24]:


#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# In[25]:


#build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax') #output layer for binary classification
])


# In[26]:


#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[27]:


#Train the model
history= model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[28]:


#evalute the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100: .2f}%")


# In[29]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


# In[30]:


#save the model for future use
model.save("brightness_detection_model.h5")


# In[31]:


#load the trained model
model = load_model("brightness_detection_model.h5")


# In[32]:


def predict_brightness(image):
    #resize the image and normalize pixel values
    image = cv2.resize(image, (64,64))/ 255.0
    #Add an extra dimension to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    
    #predict the brightness level
    prediction = model.predict(image)
    class_index= np.argmax(prediction)
    
    #map the class index to brightness level
    brightness_level = "Low Light" if class_index == 0 else "High Light"
    return brightness_level


# In[ ]:


#Initialize the webcam
cap = cv2.VideoCapture(0) # 0 is the default camera index

while True:
    #capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Filed to capture image")
        break
        
    #get brightness prediction
    brightness_level = predict_brightness(frame)
    
    #suggest action based on the prediction
   #suggestion = "Increase brightness" if brightness_level == "Low Light" else "Decrease brightness"
    if brightness_level == "Low Light":
        suggestion = "Increase brightness"
    else:
        suggestion = "Decrease brightness"
    #overlay text on the frame
    cv2.putText(frame, f"Brightness Level: {brightness_level}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Suggestion: {suggestion}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    #Display the frame with the overlay
    cv2.imshow('Camera Feed', frame)
    
    #Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#Release the camera and close all opencv windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




