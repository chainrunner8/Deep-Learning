import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
import tensorflow as tf

# Set seeds
np.random.seed(61)
tf.random.set_seed(61)

# LOAD MNIST DATA
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
print(x_train_full.shape)
print(x_train_full.dtype)

# Normalize and validation set

x_train,x_val = x_train_full[:54000]/ 255.0,x_train_full[54000:]/ 255.0
y_train,y_val = y_train_full[:54000],y_train_full[54000:]
x_test = x_test / 255.0

# CNN data
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_val_cnn = x_val.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)


# LOAD CIFAR10 DATA

(x_train_full_cifar, y_train_full_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()

# Normalize and validation set
x_train_cifar,x_val_cifar = x_train_full_cifar[:45000]/ 255.0,x_train_full_cifar[45000:]/ 255.0
y_train_cifar,y_val_cifar = y_train_full_cifar[:45000].squeeze(),y_train_full_cifar[45000:].squeeze()
x_test_cifar = x_test_cifar / 255.0
y_test_cifar = y_test_cifar.squeeze()


# ============================================================================
# MLP MODELS
# ============================================================================


def simple_mlp(x_train, y_train, x_val, y_val, x_test, y_test,model_name="Simple MLP"):

   model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(256, activation="relu",name='hidden_layer_1'),
        keras.layers.Dense(128, activation="relu", name='hidden_layer_2'),
        keras.layers.Dense(10, activation="softmax", name='output_layer')
        ])
   
   # Model summary
   model.summary()

   # Compile the model
   model.compile(
       loss='sparse_categorical_crossentropy',  
       optimizer='sgd',  
       metrics=['accuracy']
   )
   
   #Start the training
   print("\nTraining Started")
   history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
   
   
   # Evaluate on test set
   print(f"Evaluating: {model_name} on Test Set")
   test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
   
   
 
   #PRINTING RESULTS
   
   final_train_acc = history.history['accuracy'][-1]
   final_train_loss = history.history['loss'][-1]
   final_val_acc = history.history['val_accuracy'][-1]
   final_val_loss = history.history['val_loss'][-1]
   gap = final_train_acc - final_val_acc
   epochs_trained = len(history.history['loss'])
   

   print(f"RESULTS: {model_name}")
   print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
   print(f"Training Loss:          {final_train_loss:.3f}")
   print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
   print(f"Validation Loss:        {final_val_loss:.3f}")
   print(f"Test Accuracy:          {test_acc*100:.3f}%")
   print(f"Test Loss:              {test_loss:.3f}")
   print(f"Train-Val Gap:          {gap*100:.3f}%")
   print(f"Epochs Trained:         {epochs_trained}")

   
  #Plots
  
   pd.DataFrame(history.history).plot(figsize=(8, 5))
   plt.title(f'{model_name} - Learning Curves')
   plt.grid(True)
   plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
   plt.xlabel('Epoch')
   plt.ylabel('Metrics')
   plt.show()
  
   return model, history, test_loss, test_acc


## CHANGE OF REGULARIZATIONS 

# MLP WITH DROPOUT

def mlp_with_dropout(x_train, y_train, x_val, y_val, x_test, y_test, model_name="MLP with Dropout"):
    

    print(f"Training: {model_name}")
  

    model = keras.models.Sequential([
         keras.layers.Flatten(input_shape=[28, 28]),
         keras.layers.Dense(256, activation="relu",name='hidden_layer_1'),
         keras.layers.Dropout(0.3), 
         keras.layers.Dense(128, activation="relu", name='hidden_layer_2'),
         keras.layers.Dropout(0.3), 
         keras.layers.Dense(10, activation="softmax", name='output_layer')
         ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='sgd',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print("Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
   
    return model, history, test_loss, test_acc

# MLP WITH L2

def mlp_with_l2(x_train, y_train, x_val, y_val, x_test, y_test, model_name="MLP with L2"):
    

    print(f"Training: {model_name}")
  

    model = keras.models.Sequential([
         keras.layers.Flatten(input_shape=[28, 28]),
         keras.layers.Dense(256, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001),name='hidden_layer_1'), 
         keras.layers.Dense(128, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001), name='hidden_layer_2'),
         keras.layers.Dense(10, activation="softmax", name='output_layer')
         ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='sgd',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
   
    return model, history, test_loss, test_acc


## CHANGE OF OPTIMIZER

# Adam optimizer

def mlp_adam_optimizer(x_train, y_train, x_val, y_val, x_test, y_test,model_name="MLP with Adam Optimizer"):

   model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(256, activation="relu",name='hidden_layer_1'),
        keras.layers.Dense(128, activation="relu", name='hidden_layer_2'),
        keras.layers.Dense(10, activation="softmax", name='output_layer')
        ])
   
   # Model summary
   model.summary()

   # Compile the model
   model.compile(
       loss='sparse_categorical_crossentropy',  
       optimizer='adam',  
       metrics=['accuracy']
   )
   
   #Start the training
   print("\nTraining Started")
   history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
   
   
   # Evaluate on test set
   print(f"Evaluating: {model_name} on Test Set")
   test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
   
   
 
   #PRINTING RESULTS
   
   final_train_acc = history.history['accuracy'][-1]
   final_train_loss = history.history['loss'][-1]
   final_val_acc = history.history['val_accuracy'][-1]
   final_val_loss = history.history['val_loss'][-1]
   gap = final_train_acc - final_val_acc
   epochs_trained = len(history.history['loss'])
   
   print(f"RESULTS: {model_name}")
   print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
   print(f"Training Loss:          {final_train_loss:.3f}")
   print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
   print(f"Validation Loss:        {final_val_loss:.3f}")
   print(f"Test Accuracy:          {test_acc*100:.3f}%")
   print(f"Test Loss:              {test_loss:.3f}")
   print(f"Train-Val Gap:          {gap*100:.3f}%")
   print(f"Epochs Trained:         {epochs_trained}")

   
  #Plots
  
   pd.DataFrame(history.history).plot(figsize=(8, 5))
   plt.title(f'{model_name} - Learning Curves')
   plt.grid(True)
   plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
   plt.xlabel('Epoch')
   plt.ylabel('Metrics')
   plt.show()
  
   return model, history, test_loss, test_acc

# ============================================================================
# CNN MODELS
# ============================================================================

#Simple CNN

def simple_cnn(x_train, y_train, x_val, y_val, x_test, y_test, model_name="Simple CNN"):
    """Simple CNN without dropout"""
    
    print(f"Training: {model_name}")

    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='sgd',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
  
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

   
  #Plots
  
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
   
    return model, history, test_loss, test_acc


## CHANGE of OPtimizer

# CNN with Adam optimizer

def cnn_with_adam_optimizer(x_train, y_train, x_val, y_val, x_test, y_test, model_name="CNN with Adam Optimizer"):
    

    print(f"Training: {model_name}")
 
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='adam',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
  
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
    
    return model, history, test_loss, test_acc



## CHANGE of ACTIVATION

#Simple CNN with LeakyReLU 

def cnn_with_leaky_relu(x_train, y_train, x_val, y_val, x_test, y_test, model_name="Simple with LeakyReLU "):
   
    print(f"Training: {model_name}")
  
    
    model = keras.Sequential([
        
        keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        keras.layers.LeakyReLU(alpha=0.01), 
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='sgd',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
  
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
    
    return model, history, test_loss, test_acc

## CHANGE OF REGULARIZATIONS 

# CNN with Dropout

def cnn_with_dropout(x_train, y_train, x_val, y_val, x_test, y_test, model_name="CNN with Dropout"):

   
    print(f"Training: {model_name}")
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='sgd',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
  
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
    
    return model, history, test_loss, test_acc

def deeper_cnn(x_train, y_train, x_val, y_val, x_test, y_test, model_name="Deeper CNN"):
  
    print(f"Training: {model_name}")
   
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='sgd',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
  
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
    
    return model, history, test_loss, test_acc


# ============================================================================
# CIFAR MODELS
# ============================================================================

def mlp_adam_optimizer_cifar(x_train, y_train, x_val, y_val, x_test, y_test,model_name="CIFAR-10 MLP with Adam Optimizer"):

   model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(256, activation="relu",name='hidden_layer_1'),
        keras.layers.Dense(128, activation="relu", name='hidden_layer_2'),
        keras.layers.Dense(10, activation="softmax", name='output_layer')
        ])
   
   # Model summary
   model.summary()

   # Compile the model
   model.compile(
       loss='sparse_categorical_crossentropy',  
       optimizer='adam',  
       metrics=['accuracy']
   )
   
   #Start the training
   print("\nTraining Started")
   history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
   
   
   # Evaluate on test set
   print(f"Evaluating: {model_name} on Test Set")
   test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
   
   
 
   #PRINTING RESULTS
   
   final_train_acc = history.history['accuracy'][-1]
   final_train_loss = history.history['loss'][-1]
   final_val_acc = history.history['val_accuracy'][-1]
   final_val_loss = history.history['val_loss'][-1]
   gap = final_train_acc - final_val_acc
   epochs_trained = len(history.history['loss'])
   
   print(f"RESULTS: {model_name}")
   print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
   print(f"Training Loss:          {final_train_loss:.3f}")
   print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
   print(f"Validation Loss:        {final_val_loss:.3f}")
   print(f"Test Accuracy:          {test_acc*100:.3f}%")
   print(f"Test Loss:              {test_loss:.3f}")
   print(f"Train-Val Gap:          {gap*100:.3f}%")
   print(f"Epochs Trained:         {epochs_trained}")

   
  #Plots
  
   pd.DataFrame(history.history).plot(figsize=(8, 5))
   plt.title(f'{model_name} - Learning Curves')
   plt.grid(True)
   plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
   plt.xlabel('Epoch')
   plt.ylabel('Metrics')
   plt.show()
  
   return model, history, test_loss, test_acc

def simple_cnn_cifar(x_train, y_train, x_val, y_val, x_test, y_test, model_name="CIFAR 10 Simple CNN"):
   
    
    print(f"Training: {model_name}")

    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='sgd',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
  
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
    
    return model, history, test_loss, test_acc


def cnn_with_adam_optimizer_cifar(x_train, y_train, x_val, y_val, x_test, y_test, model_name="CIFAR-10 with CNN Adam Optimizer "):

 
    print(f"Training: {model_name}")
    
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
       
    # Model summary
    model.summary()

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  
        optimizer='adam',  
        metrics=['accuracy']
    )
    
    #Start the training
    print("\nTraining Started")
    history = model.fit(x_train, y_train,batch_size=128, epochs=30,validation_data=(x_val, y_val), verbose=1 )
    
    
    # Evaluate on test set
    print(f"Evaluating: {model_name} on Test Set")
    test_loss, test_acc= model.evaluate(x_test, y_test, verbose=0)
    
    
  
    #PRINTING RESULTS
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    gap = final_train_acc - final_val_acc
    epochs_trained = len(history.history['loss'])
    
    print(f"RESULTS: {model_name}")
    print(f"Training Accuracy:      {final_train_acc*100:.3f}%")
    print(f"Training Loss:          {final_train_loss:.3f}")
    print(f"Validation Accuracy:    {final_val_acc*100:.3f}%")
    print(f"Validation Loss:        {final_val_loss:.3f}")
    print(f"Test Accuracy:          {test_acc*100:.3f}%")
    print(f"Test Loss:              {test_loss:.3f}")
    print(f"Train-Val Gap:          {gap*100:.3f}%")
    print(f"Epochs Trained:         {epochs_trained}")

    
   #Plots
   
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'{model_name} - Learning Curves')
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.show()
    
    return model, history, test_loss, test_acc



# RUN EXPERIMENTS
def run_experiments():
 
  
    print("FASHION MNIST EXPERIMENTS")
    
    
    results = []
    
    # Experiment 0: CNN with Adam Optimizer
    model, history, test_loss, test_acc = cnn_with_adam_optimizer(
        x_train_cnn, y_train, x_val_cnn, y_val, x_test_cnn, y_test,
        model_name="CNN with Adam Optimizer"
    )
    results.append(("CNN with Adam Optimizer", test_acc))
    
    # Experiment 1: Simple MLP
    model, history, test_loss, test_acc = simple_mlp(
        x_train, y_train, x_val, y_val, x_test, y_test,
        model_name="Simple MLP"
    )
    results.append(("Simple MLP", test_acc))
    
    # Experiment 2: MLP with Dropout
    model, history, test_loss, test_acc = mlp_with_dropout(
        x_train, y_train, x_val, y_val, x_test, y_test,
        model_name="MLP with Dropout"
    )
    results.append(("MLP with Dropout", test_acc))
    
    # Experiment 3: MLP with L2
    model, history, test_loss, test_acc = mlp_with_l2(
        x_train, y_train, x_val, y_val, x_test, y_test,
        model_name="MLP with L2"
    )
    results.append(("MLP with L2", test_acc))
    
    # Experiment 4: MLP with Adam Optimizer
    model, history, test_loss, test_acc = mlp_adam_optimizer(
        x_train, y_train, x_val, y_val, x_test, y_test,
        model_name="MLP with Adam Optimizer"
    )
    results.append(("MLP with Adam Optimizer", test_acc))
    
    # Experiment 5: Simple CNN 
    model, history, test_loss, test_acc = simple_cnn(
        x_train_cnn, y_train, x_val_cnn, y_val, x_test_cnn, y_test,
        model_name="Simple CNN"
    )
    results.append(("Simple CNN", test_acc))
    
    # Experiment 6: CNN with LeakyRelu
    model, history, test_loss, test_acc = cnn_with_leaky_relu(
        x_train_cnn, y_train, x_val_cnn, y_val, x_test_cnn, y_test,
        model_name="CNN with LeakyRelu"
    )
    results.append(("CNN with LeakyRelu", test_acc))
    
    # Experiment 7: CNN with Dropout
    model, history, test_loss, test_acc = cnn_with_dropout(
        x_train_cnn, y_train, x_val_cnn, y_val, x_test_cnn, y_test,
        model_name="CNN with Dropout"
    )
    results.append(("CNN with Dropout", test_acc))
    
    # Experiment 8: Deeper CNN
    model, history, test_loss, test_acc = deeper_cnn(
        x_train_cnn, y_train, x_val_cnn, y_val, x_test_cnn, y_test,
        model_name="Deeper CNN"
    )
    results.append(("Deeper CNN", test_acc))
    
    print("CIFAR-10 EXPERIMENTS")
    
    
    # Experiment 9: CIFAR-10 with CNN Adam Optimizer       
    model, history, test_loss, test_acc = cnn_with_adam_optimizer_cifar(
        x_train_cifar, y_train_cifar, x_val_cifar, y_val_cifar, x_test_cifar, y_test_cifar,
        model_name="CIFAR-10 with CNN Adam Optimizer "
    )
    results.append(("CIFAR-10 with CNN Adam Optimizer", test_acc))
    
    # Experiment 10: CIFAR-10 with MLP Adam Optimizer
    model, history, test_loss, test_acc = mlp_adam_optimizer_cifar(
        x_train_cifar, y_train_cifar, x_val_cifar, y_val_cifar, x_test_cifar, y_test_cifar,
        model_name="CIFAR-10 with MLP Adam Optimizer"
    )
    results.append(("CIFAR-10 with MLP Adam Optimizer", test_acc))
    
    # Experiment 11: CIFAR-10 with Simple CNN
    model, history, test_loss, test_acc = simple_cnn_cifar(
        x_train_cifar, y_train_cifar, x_val_cifar, y_val_cifar, x_test_cifar, y_test_cifar,
        model_name="CIFAR-10 with Simple CNN"
    )
    results.append(("CIFAR-10 with Simple CNN", test_acc))
    


    # Final summary
  
    print("FINAL ALL RESULTS")

    for name, acc in results:
        print(f"{name:30s}: {acc*100:.2f}%")
    print("="*60)
    
    return results


if __name__ == '__main__':
    
    # Run experiments
    results = run_experiments()
    
    print("Experiments done")