import time
import logging
import os
#suppressing the huge logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR) 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.compat.v1.keras.backend import set_session

from dataset.data_loader import get_data, reset_Models
from util.augmentors import mixup, cutmix, cutout, single_cutout

# needed to make gpu work
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#seeds
#seed = 1234
#np.random.seed(seed)
#tf.random.set_seed(seed)
#tf.random.uniform([1], seed=seed)  # doesn't work?

def setup_model(aug1,aug2):
    labels = ['NORMAL', 'PNEUMONIA']

    # Define parameters for our network
    batch_size = 128
    epochs = 50
    img_height = 150
    img_width = 150
    img_dims = (img_height, img_width)
    alpha = 1
    num_holes = 5

    reset_Models()
    process_data = False
    (train_x0, train_y0), (val_x0, val_y0), (test_x0, test_y0) = get_data(img_dims, labels,process_data)#(val_x, val_y), (test_x, test_y) = get_data(img_dims, labels,process_data)
    total_train = len(train_x0)
    total_val = len(val_x0)
    total_test = len(test_x0)
    total = total_train+total_val+total_test
    print('\n#images for training:{} ({}%)'.format(total_train, round(total_train/total*100,1)),
            ' validation:{} ({}%)'.format(total_val, round(total_val/total*100,1)),
            ' testing:{} ({}%)'.format(total_test, round(total_test/total*100,1)),
            ' total: {}'.format(total))

    losses = []
    scores = []
    hist = []

    for i in range(15):
        train_x = train_x0.copy()
        train_y = train_y0.copy()
        val_x = val_x0.copy()
        val_y = val_y0.copy()
        test_x = test_x0.copy()
        test_y = test_y0.copy()



        if aug1 == 'None':
            gen = ImageDataGenerator()
        elif aug1 == 'Basic':
            gen = ImageDataGenerator(
                rotation_range=5,
                zoom_range=0.2,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )
        elif aug1 == 'Mixup':
            train_x, train_y = mixup(train_x, train_y, alpha, show_sample=False)
            gen = ImageDataGenerator()
        elif aug1 == 'Cutmix':
            train_x, train_y = cutmix(train_x, train_y, alpha, show_sample=False)
            gen = ImageDataGenerator()
        elif aug1 == 'Cutout':
            train_x, train_y = cutout(train_x, train_y, n_holes=num_holes, show_sample=False)
            gen = ImageDataGenerator()
            

        # if aug1 != aug2:
        #     if aug2 == 'Mixup':
        #         train_x, train_y = mixup(train_x, train_y, alpha, show_sample=False)
        #     elif aug2 == 'Cutmix':
        #         train_x, train_y = cutmix(train_x, train_y, alpha, show_sample=False)
        #     elif aug2 == 'Cutout':
        #         train_x, train_y = cutout(train_x, train_y, n_holes=num_holes, show_sample=False)

        #only need to fit the generator if featurewise_center or featurewise_std_normalization or zca_whitening are set to True
        #gen.fit(train_x, seed=seed)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.00001, cooldown=3, verbose=1)



        train_gen = gen.flow(train_x, train_y, batch_size,shuffle=True)#, seed=seed)
        val_gen = gen.flow(val_x, val_y, batch_size)#, seed=seed)

        #neural network setup
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu',
                input_shape=(img_height, img_width, 1)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1)
        ])

        #prints a summary of the model
        #model.summary()
        
        #saves weights
        path = 'saved_models/{}_weights'.format(aug1+aug2)

        # model.load_weights(path) # uncomment to load weights from last checkpoint
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath=path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

        history = model.fit(
            train_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=total_val // batch_size,
            shuffle=False, 
            callbacks=[reduce_lr, checkpoint]

        )
        model.load_weights(path)  # load weights for best epoch

        test_loss, test_score = model.evaluate(test_x, test_y, batch_size=batch_size)
        #print("Loss on test set: ", test_loss)
        #print("Accuracy on test set: ", test_score)

        saveAugPlot = aug1+aug2+' '+str(i)
        plot_model(history,saveAugPlot)
        losses.append(test_loss)    
        scores.append(test_score)
        hist.append(history)

        print('\nFinished iteration #{} of {} and {}\n'.format(i+1,aug1,aug2))
    
    plot_model(hist,(aug1+aug2+' Average'))
    return losses,scores 


def plot_model(results,saveAugPlot):
    if 'Average' not in saveAugPlot:

        acc = results.history['accuracy']
        val_acc = results.history['val_accuracy']

        loss = results.history['loss']
        val_loss = results.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(fname=('plots/'+saveAugPlot))

    else:
        acc, val_acc, loss, val_loss = [], [], [], []
        for hist in results:
            acc.append(hist.history['accuracy'])
            val_acc.append(hist.history['val_accuracy'])

            loss.append(hist.history['loss'])
            val_loss.append(hist.history['val_loss'])


        avgAcc = np.average(acc,axis=0)
        avgValAcc = np.average(val_acc,axis=0)
        avgLoss = np.average(loss,axis=0)
        avgValLoss = np.average(val_loss,axis=0)
        
        epochs_range = range(len(avgAcc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, avgAcc, label='Training Accuracy')
        plt.plot(epochs_range, avgValAcc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, avgLoss, label='Training Loss')
        plt.plot(epochs_range, avgValLoss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(fname=('plots/'+saveAugPlot))


    plt.close('all')


def runAll():
    augmentations = ['None','Basic','Cutmix','Mixup','Cutout']
    resLosses = []
    resScores = []
    with open('results.txt','w') as file:
        # for i in range(len(augmentations)):
        #     for j in range(len(augmentations)):
        #         tmp = setup_model(augmentations[i],augmentations[j])
        #         resLosses.append(tmp[0])
        #         resScores.append(tmp[1])
        #         file.write('\nAugmentations: {} {}'.format(augmentations[i],augmentations[j]))
        #         file.write('\nLosses: {}'.format(tmp[0]))
        #         file.write('\nScores: {}'.format(tmp[1]))
        #         file.write('\nAverage Loss: {}'.format(np.average(tmp[0])))
        #         file.write('\nAverage Score: {}'.format(np.average(tmp[1])))

        tmp = setup_model(augmentations[4],augmentations[4])
        resLosses.append(tmp[0])
        resScores.append(tmp[1])
        file.write('\nAugmentations: {} {}'.format(augmentations[4],augmentations[4]))
        file.write('\nLosses: {}'.format(tmp[0]))
        file.write('\nScores: {}'.format(tmp[1]))
        file.write('\nAverage Loss: {}'.format(np.average(tmp[0])))
        file.write('\nAverage Score: {}'.format(np.average(tmp[1])))


    #print(resLosses)
    #print(resScores)
   

if __name__ == '__main__':
    startTime = time.time()
    runAll()
    endTime = time.time()
    hours, rem = divmod(endTime-startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Elapsed time: {:0>2}:{:0>2}:{:0>2}'.format(int(hours),int(minutes),int(seconds)))

