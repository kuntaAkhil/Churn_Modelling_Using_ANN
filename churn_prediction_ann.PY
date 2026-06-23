import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,ReLU,Dropout

# Load the dataset
Dataset=pd.read_csv('Churn_Modelling.csv')

# Separate input features and output label
X=Dataset.iloc[:, 3:13]
Y=Dataset.iloc[:, 13]

# Convert categorical columns into numeric values
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

# Remove original categorical columns and add encoded columns
X=X.drop(['Geography','Gender'],axis=1)
X=pd.concat([X,geography,gender],axis=1)

# Split data into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Scale the input features
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Create the ANN model
classifier=Sequential()

# Add hidden and output layers
classifier.add(Dense(units=11,activation='relu'))
classifier.add(Dense(units=7,activation='relu'))
classifier.add(Dense(units=6,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

# Compile the model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Stop training if validation loss does not improve
early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

# Train the model
model_history=classifier.fit(
    X_train,
    Y_train,
    validation_split=0.33,
    batch_size=10,
    epochs=1000,
    callbacks=early_stopping
)

# Print the available training metrics
print(model_history.history.keys())

# Plot training and validation accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make predictions on the test data
y_pred=classifier.predict(X_test)

# Convert probabilities into 0 or 1
y_pred=(y_pred >= 0.5)

# Generate the confusion matrix
cm=confusion_matrix(Y_test,y_pred)

print(cm)

# Calculate the model accuracy
score=accuracy_score(y_pred,Y_test)

print(f"The accurascy score is {score}")