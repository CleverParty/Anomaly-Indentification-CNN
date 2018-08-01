from __future__ import division
import pdb
import numpy as np
import matplotlib.pyplot as plt
import math, cv2, os, shutil
from datetime import datetime
import time, timeit
import pickle
import bcolz

from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras import optimizers
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras import regularizers

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from PIL import Image
from PIL import ImageFile

# the following fixes an error in image processing - (via stackoverflow)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def fpath(dirname, filename):
    return os.path.join(dirname, filename)


def define_paths(base_dir_path, project, model_used, trial_num):
    global base_dir, train_data_dir, validation_data_dir, model_save_path
    global top_model_path, top_best_model_path, full_model_path, full_best_model_path
    global top_model_save_path

    base_dir = base_dir_path
    train_data_dir = fpath(base_dir, 'training')
    validation_data_dir = fpath(base_dir, 'validation')
    model_save_path = fpath(base_dir, 'models')
    project_fname = project + '-' + model_used + '-' + 'trial-' + str(trial_num) + '.h5'
    top_model_path = fpath(model_save_path, 'top-' + project_fname)
    top_best_model_path = fpath(model_save_path, 'top-best-' + project_fname)
    full_model_path = fpath(model_save_path, 'full-' + project_fname)
    full_best_model_path = fpath(model_save_path, 'full-best-' + project_fname)
    top_train_features = fpath(model_save_path, 'top_train_features,bc')
    top_validation_features = fpath(model_save_path, 'top_validation_features,bc')
    top_model_save_path = fpath(model_save_path, 'top_model.h5')


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)


def plot(img):
    plt.imshow(to_plot(img))


def step_decay(epoch):
    initial_lrate = initial_learning_rate
    decay_rate = 0.95
    num_epochs_to_change_rate = 3.0
    # effective rate =  intial_rate* (decay_rate ^ (1+epoch/num_epochs_to_change_rate))
    lrate = initial_lrate * math.pow(decay_rate, math.floor((1 + epoch) / num_epochs_to_change_rate))
    print("Learning rate set to..", lrate)
    return lrate


def find_last_conv(model, num):
    i = 0
    layer_nums = []
    for layer in model.layers:
        # print(i,layer.__class__,layer.name,"Conv" if layer.__class__.__name__ == "Conv2D" else "Non-Conv")
        if layer.__class__.__name__ == "Conv2D":
            layer_nums.append(i)
        i += 1
    # layer_nums is a list of all the Conv2D layer numbers     
    # if num = 1 then layer_nums[-1] will be the number of the last conv layer   
    layer_num = layer_nums[-num]
    return layer_num


def set_conv_trainable(model, num_conv_layers_to_train):
    first_trainable_layer = find_last_conv(model, num_conv_layers_to_train)
    for layer in model.layers[:first_trainable_layer]:
        layer.trainable = False
    for layer in model.layers[first_trainable_layer:]:
        layer.trainable = True
    print("Setting conv layers from ", first_trainable_layer, " as trainable ")


def compile_model(model, optimizer='adam'):
    # Deafaults for Adam
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0True)rad=False)
    if optimizer == 'adam':
        opt = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    if optimizer == 'sgd':
        opt = SGD(lr=initial_learning_rate, momentum=0.9)
    if optimizer == 'rmsprop':
        opt = RMSprop(lr=initial_learning_rate, rho=0.9, epsilon=None, decay=0.0)
    print("Now compiling ", opt)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(opt, " Compiled")


def get_pretrained_model(pretrained_model):
    if pretrained_model == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
    if pretrained_model == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    if pretrained_model == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    if pretrained_model == 'inceptionv3r2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    return base_model


def run_predictions(predict_dir, predict_model):
    # run predictions on a model - returns the probabilities, and the prediction generator
    model = load_model(predict_model)
    predict_datagen = ImageDataGenerator(rescale=1. / 255)
    predict_gen = predict_datagen.flow_from_directory(
        predict_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None, shuffle=False)

    probs = model.predict_generator(predict_gen, verbose=1)
    return probs, predict_gen


def predict_images(predict_dir, predict_model, class_to_predict=0):
    # predict_dir = '/media/arun/data/wellcare/xsimple1/training'
    probs, predict_gen = run_predictions(predict_dir, predict_model)

    where_class_true = np.argmax(probs, axis=1) == class_to_predict  # returns a boolean array where class occurs

    predict_list = []
    for i in range(len(probs)):
        # for each index i where class is true find the corresponding filename returned in predict_gen
        if where_class_true[i]:
            filename = os.path.join(predict_dir, predict_gen.filenames[i])
            # display_image(filename)
            predict_list.append(filename)

    return predict_list


def display_predictions_mini(predict_list):
    fn = len(predict_list)

    cols = 2
    rows = fn // cols if fn % 2 == 0 else fn // cols + 1

    # x = range(10)
    # y = range(10)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(40, 40))
    plt.subplots_adjust(wspace=0.05, hspace=0.70)

    i = 0
    for row in ax:
        for col in row:
            if i < fn:
                image = load_img(predict_list[i])
                image = img_to_array(image)
                col.imshow(to_plot(image))
                head, fname = os.path.split(predict_list[i])
                col.set_title(head + '\n' + fname, size=30)

            i += 1

    plt.show()


def display_predictions(predict_list):
    for img in predict_list:
        image = load_img(img)
        image = img_to_array(image)
        head, fname = os.path.split(img)
        plt.title(head + '\n' + fname)
        plt.imshow(to_plot(image))
        plt.show()


def gen_samples(data_dir):
    print("Entering gen_samples....")
    print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ", K.image_data_format())

    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print("Number of Samples = ", len(generator.filenames))
    print("Classes Found = ", generator.class_indices)
    print("Number of Classes = ", len(generator.class_indices))

    num_samples = generator.n
    num_classes = len(generator.class_indices)

    num_steps = int(math.ceil(num_samples / batch_size))
    print("Number of Training Steps (Num samples / Batch Size) =  ", num_steps)

    data_dict = dict()
    data_dict['num_samples'] = generator.n
    data_dict['num_classes'] = len(generator.class_indices)
    data_dict['class_indices'] = generator.class_indices
    data_dict['classes'] = generator.classes
    data_dict['filenames'] = generator.filenames
    data_dict['num_steps'] = num_steps

    return generator, data_dict


def gen_train_batch(X, Y, batch_size, num_samples):
    # randomseed = 0
    # np.random.seed(randomseed)
    # np.random.shuffle(X)
    # np.random.seed(randomseed)
    # np.random.shuffle(Y)
    i = 0
    print("Generating training batch data")

    while 1:
        # randomseed += 1
        if (i + 1) * batch_size < num_samples:
            # print("\nTrain Step No. = ",i,"Y =",Y[i*batch_size:i*batch_size + 8,:] )
            yield X[i * batch_size:(i + 1) * batch_size, :], Y[i * batch_size:(i + 1) * batch_size, :]

            i += 1
        else:
            # print("\n Train i= ",i,"seed = ",randomseed,"i*batch_size = ",i*batch_size,"Y ",Y[i*batch_size:num_samples,:])
            # print("\nLast Train Step = ",i,"Y =",Y[i*batch_size:(i+1)*batch_size,:] )
            yield X[i * batch_size:num_samples, :], Y[i * batch_size:num_samples, :]
            i = 0
            # np.random.seed(randomseed)
            # np.random.shuffle(X)
            # np.random.seed(randomseed)
            # np.random.shuffle(Y)
            # print("\n End Train i= ",i,"seed = ",randomseed)


def gen_val_batch(X, Y, batch_size, num_samples):
    # randomseed = 0
    # np.random.seed(randomseed)
    # np.random.shuffle(X)
    # np.random.seed(randomseed)
    # np.random.shuffle(Y)
    print("Generating validation batch data")
    i = 0

    # if debug_mode:
    #    print("Entering debug Mode...validation batch - first entry..")
    #    pdb.set_trace()

    while 1:
        # randomseed += 1

        if (i + 1) * batch_size < num_samples:

            # print("\nVal Step = ",i,"Y =",Y[i*batch_size:i*batch_size + 8,:])
            yield X[i * batch_size:(i + 1) * batch_size, :], Y[i * batch_size:(i + 1) * batch_size, :]
            i += 1
        else:
            # if debug_mode:
            #    print("Entering debug Mode...validation batch - last entry..")
            #    pdb.set_trace()
            # print("\nLast Val Step = ",i,"Y =",Y[i*batch_size:(i+1)*batch_size,:])
            # print("\nVal i= ",i,"seed = ",randomseed,"i*batch_size = ",i*batch_size,"Y ",Y[i*batch_size:num_samples,:])
            yield X[i * batch_size:num_samples, :], Y[i * batch_size:num_samples, :]
            i = 0
            # np.random.seed(randomseed)
            # np.random.shuffle(X)
            # np.random.seed(randomseed)
            # np.random.shuffle(Y)
            # print("\n End Valid i= ",i,"seed = ",randomseed)


def train_bottleneck_model(pre_trained_model='vgg16', load_saved_training_data=False, load_saved_model=False,
                           train_top_using_batches=False):
    print("Entering train_bottleneck_model..")
    print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ", K.image_data_format())

    if load_saved_training_data:

        print("Loading training data ")
        train_data = load_array(fpath(model_save_path, 'train-data.bc'))
        print("Loading training labels ")
        train_labels = load_array(fpath(model_save_path, 'train-labels.bc'))
        print("Loading validation data ")
        validation_data = load_array(fpath(model_save_path, 'validation-data.bc'))
        print("Loading validation labels ")
        validation_labels = load_array(fpath(model_save_path, 'validation-labels.bc'))

        data_dict = np.load(fpath(model_save_path, 'train_data_dict.npy')).item()
        num_train_samples = data_dict['num_samples']
        num_steps = data_dict['num_steps']
        num_classes = data_dict['num_classes']

        data_dict = np.load(fpath(model_save_path, 'validation_data_dict.npy')).item()
        num_val_steps = data_dict['num_steps']
        num_valid_samples = data_dict['num_samples']
        if debug_mode:
            print("Loading saved data..train-data is..", train_data[:5])
            pdb.set_trace()
    else:

        print("Now  generating training data.. ")
        train_gen, data_dict = gen_samples(train_data_dir)

        num_train_samples = data_dict['num_samples']
        num_classes = data_dict['num_classes']
        num_steps = data_dict['num_steps']
        train_labels = train_gen.classes
        if debug_mode:
            print("Read training data...check the sequence  using train_gen", train_labels[:5])
            pdb.set_trace()

        train_labels = to_categorical(train_labels, num_classes=num_classes)

        np.save(fpath(model_save_path, 'train_data_dict.npy'), data_dict)

        print("Generating bottleneck predictions..")
        base_model = get_pretrained_model(pretrained_model=pre_trained_model)
        train_data = base_model.predict_generator(train_gen, verbose=1)
        if debug_mode:
            print("Basemodel.predict is done..this is the train_data..", train_data[:5])
            pdb.set_trace()

        save_array(fpath(model_save_path, 'train-data.bc'), train_data)
        save_array(fpath(model_save_path, 'train-labels.bc'), train_labels)

        print("Generating validation data...")
        # pdb.set_trace()
        validation_gen, data_dict = gen_samples(validation_data_dir)

        num_valid_samples = data_dict['num_samples']
        num_val_steps = data_dict['num_steps']
        validation_labels = validation_gen.classes
        validation_labels = to_categorical(validation_labels, num_classes=num_classes)
        np.save(fpath(model_save_path, 'validation_data_dict.npy'), data_dict)

        validation_data = base_model.predict_generator(validation_gen, verbose=1)
        save_array(fpath(model_save_path, 'validation-data.bc'), validation_data)
        save_array(fpath(model_save_path, 'validation-labels.bc'), validation_labels)
        if debug_mode:
            print("Basemodel.predict  done for validation data..this is the data", validation_labels[:5])
            pdb.set_trace()

            # if debug_mode:
    # print("Entering debug Mode...training data loaded..")
    #    pdb.set_trace()  

    if load_saved_model:

        # 1model = load_model(top_model_save_path)
        model = load_model(top_best_model_path)


    else:
        print("Creating the full model...")
        # now create the full model
        inputs = Input(shape=train_data.shape[1:])
        x = Flatten(input_shape=train_data.shape[1:])(inputs)
        x = GlobalAveragePooling2D()(inputs)

        x = Dense(dense_layer_neurons, activation='relu')(x)
        # x  = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout)(x)

        x = Dense(dense_layer_neurons, activation='relu')(x)
        # x  = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout)(x)

        x = Dense(dense_layer_neurons, activation='relu')(x)
        # x  = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout)(x)

        x = Dense(dense_layer_neurons, activation='relu')(x)
        # x  = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout)(x)

        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)

        # if train_conv_layers:    set_conv_trainable(model,num_conv_layers_to_train)

        compile_model(model, optimizer=optimizer_used)

        print("Model compiled...")

        # dave the top model for later
        model.save(top_model_save_path)

    checkpoint = ModelCheckpoint(filepath=top_best_model_path, monitor='val_acc', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')
    lrate = LearningRateScheduler(step_decay, verbose=1)
    callback_list = [checkpoint]
    if use_learning_decay:          callback_list = callback[checkpoiint, lrate]
    if early_stopping:              callback_list = [checkpoint, earlystop]

    # pdb.set_trace()
    print("Starting Training...")

    if train_top_using_batches:
        train_gen_bneck = gen_train_batch(train_data, train_labels, batch_size, num_train_samples)
        valid_gen_bneck = gen_val_batch(validation_data, validation_labels, batch_size, num_valid_samples)
        print("Training using batches")
        if debug_mode:
            print("Entering debug Mode...train_gen_neck")
            pdb.set_trace()

        history = model.fit_generator(
            train_gen_bneck,
            steps_per_epoch=num_steps,
            epochs=epochs,
            validation_data=valid_gen_bneck,
            validation_steps=num_val_steps, callbacks=callback_list)

        if debug_mode:
            print("Entering debug Mode...training done..")
            pdb.set_trace()

    else:

        history = model.fit(x=train_data, y=train_labels,
                            steps_per_epoch=num_steps,
                            epochs=epochs,
                            validation_data=(validation_data, validation_labels),
                            validation_steps=num_val_steps, callbacks=callback_list)

    elapsed = timeit.default_timer() - start_time

    print("Total Time Elapsed for Training = {0:.2f} ".format(elapsed))

    print("Saving Final Model ")

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    return model, history


# modes =  reg_train, top_train, predict, find_invalid   '



def load_archive():
    try:
        model_archives = np.load(os.path.join(model_save_path, 'model_archives.npy')).item()

    except FileNotFoundError:
        model_archives = dict()

    return model_archives


def move_files(predict_base_dir, folder):
    predict_list_full = np.load(fpath(predict_base_dir, 'predict_filenames.npy'))
    print("Now moving matching files...")

    dest_folder = fpath(predict_base_dir, folder)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    i = 0
    for file_name in predict_list_full:
        if not '00A-' in file_name:
            shutil.move(file_name, dest_folder)
            i += 1
    print(i, " Files Moved to", dest_folder)


def train_full(continue_top_training=False, debug_mode=False):
    print("Entering train_mode..")
    print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ", K.image_data_format())

    class_folders = os.listdir(train_data_dir)
    numclasses = len(class_folders)
    print("Found " + str(numclasses) + " classes in the Training Folder: " + train_data_dir)

    if continue_top_training:

        model = load_model(top_best_model_path)

        if train_conv_layers:  set_conv_trainable(model, num_conv_layers_to_train)
        if recompile_model:   compile_model(model, optimizer=optimizer_used)

        if debug_mode:
            print("Entering Debug Mode... just after loading model ")
            pdb.set_trace()


    else:
        base_model = get_pretrained_model(pretrained_model)
        if debug_mode:
            print("Entering debug Mode...just after compile model")
            pdb.set_trace()

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(dense_layer_neurons, activation='relu')(x)
        if use_batch_norm: x = BatchNormalization()(x)
        x = Dropout(dense_layer_dropout)(x)
        x = Dense(dense_layer_neurons, activation='relu')(x)
        if use_batch_norm:  x = BatchNormalization()(x)
        x = Dropout(dense_layer_dropout)(x)
        x = Dense(dense_layer_neurons, activation='relu')(x)
        # x  = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout)(x)
        x = Dense(dense_layer_neurons, activation='relu')(x)
        # if use_batch_norm:  x = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(numclasses, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        if train_conv_layers:    set_conv_trainable(model, num_conv_layers_to_train)

        compile_model(model, optimizer=optimizer_used)

        print("Setting Up Image Data Generator...")
        if debug_mode:
            print("Entering debug Mode...just after compile model")
            pdb.set_trace()

    # bottleneck_features_train = base_model.predict_generator(train_generator, steps = num_steps,verbose = 1)


    if data_aug:

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.05,
            rotation_range=25,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            channel_shift_range=0.15,
            horizontal_flip=True,
            fill_mode='constant')
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)

    print("Setting up Training Generator")
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    nb_train_samples = len(train_generator.filenames)
    num_classes = len(train_generator.class_indices)

    print("Num Training Examples =  ", nb_train_samples)

    print("Setting up Validation Generator")

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    nb_validation_samples = len(validation_generator.filenames)
    print("Num Validation  Samples ", nb_validation_samples)

    classwts = {0: 4.884889059271241,
                1: 1.0,
                2: 5.860233990285925,
                3: 8.952072516664096,
                4: 6.659112346088153,
                5: 10.514077562438366,
                6: 10.726716491458607,
                7: 10.840229074158607,
                8: 10.804359515241758,
                9: 10.893346400050042}

    checkpoint = ModelCheckpoint(filepath=top_best_model_path, monitor='val_acc', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    lrate = LearningRateScheduler(step_decay, verbose=1)
    # callback_list = [checkpoint,lrate]
    if use_learning_decay:
        callback_list = [checkpoint, lrate]
    else:
        callback_list = [checkpoint]

    # if early_stopping:              callback_list = callback_list.append(earlystop)

    print("Starting Training...")
    if debug_mode:
        print("Entering debug Mode...just before Model.fit_generator")
        pdb.set_trace()

    history1 = model.fit_generator(
        train_generator,
        steps_per_epoch=int(math.ceil(nb_train_samples / batch_size)),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=int(math.ceil(nb_validation_samples / batch_size)), callbacks=callback_list)

    # class_weight = classwts,

    elapsed = timeit.default_timer() - start_time

    print("Total Time Elapsed for Training = {0:.2f} ".format(elapsed))
    print("Best Model Saved as ", top_best_model_path)

    print("Saving Final Model ")
    model.save(top_model_path)  # creates a HDF5 file 'my_model.h5'

    print("Final Model Saved as ", top_model_path)

    finish_time = time.asctime()
    print("All done at: ", finish_time)

    if save_archive:
        print(" Now saving model archive file...")
        model_archive = dict()
        model_archive["project"] = project
        model_archive["model_used"] = model_used
        model_archive["base_dir"] = base_dir
        model_archive["trial_num"] = trial_num
        model_archive["numclasses"] = numclasses
        model_archive["epochs"] = epochs
        model_archive["batch_size"] = batch_size
        model_archive["initial_learning_rate"] = initial_learning_rate
        model_archive["image_size"] = (img_height, img_width)
        model_archive["data_aug"] = data_aug
        model_archive["nb_training_samples"] = nb_train_samples
        model_archive["nb_validation_samples"] = nb_validation_samples
        model_archive["Elapsed_1"] = elapsed
        model_archive["history1"] = history1
        model_archive["model_paths"] = {"top_best_model_path": top_best_model_path,
                                        "top_model_path": top_model_path}
        model_archive["continue_top_training"] = continue_top_training
        model_archive["use_learning_decay"] = use_learning_decay
        model_archive["optimizer_used"] = optimizer_used
        model_archive["recompile_model"] = recompile_model
        model_archive["train_conv_layers"] = train_conv_layers
        model_archive["num_conv_layers_to_train"] = num_conv_layers_to_train
        model_archive["early_stopping"] = early_stopping
        model_archive["dense_layer_neurons"] = dense_layer_neurons
        model_archive["dense_layer_dropout"] = dense_layer_dropout
        model_archive["validation_accuracy"] = history1.history["val_acc"]
        model_archive["training_accuracy"] = history1.history["acc"]

        model_archive["start_time"] = start_time
        model_archive["finish_time"] = finish_time

        model_archives = load_archive()
        t = time.asctime()
        model_archives[t] = model_archive

        # if debug_mode:
        # print("Entering debug Mode...just before returning from function")
        # pdb.set_trace()

        np.save(os.path.join(model_save_path, 'model_archives.npy'), model_archives)
        print("Model archive saved..all done!")

    if debug_mode:
        print("Entering debug Mode...just before returning from function")
        pdb.set_trace()

    return model, history1


def full_model(model_path):
    # this combines the pre-trained base model and its 
    # predictions with  the top model to create the full
    # model that can then be used for predictions

    base_model = get_pretrained_model(pretrained_model)
    print("Base Model loaded..now loading top model..")
    top_model = load_model(model_path)

    print("Top model loaded...")
    if K.image_data_format() == "channels_first":
        inputshape = (3, img_height, img_width)
    else:
        inputshape = (img_height, img_width, 3)
    main_input = Input(shape=inputshape)
    main_output = base_model(main_input)

    top_output = top_model(main_output)

    full_model = Model(inputs=main_input, outputs=top_output)

    compile_model(full_model, optimizer=optimizer_used)

    full_model.save(fpath(model_save_path, 'full-model.h5'))
    # return full_model
    print("Full Model Saved")
    return fpath(model_save_path, 'full-model.h5')


def predict_mode(predict_model, predict_base_dir, predict_sub_folders=['test', 'training', 'validation'],
                 class_to_predict=0, predict_save_file='predict_filenames.npy', disp_pred=False):
    predict_dir_list = []

    # for  f in ['test','training','validation']:
    for f in predict_sub_folders:
        predict_dir_list.append(fpath(predict_base_dir, f))

    predict_list_full = []
    for predict_dir in predict_dir_list:
        predict_list = predict_images(predict_dir, predict_model, class_to_predict=class_to_predict)
        print("Found ", len(predict_list), " Matching Images in...", predict_dir)
        if disp_pred:  display_predictions(predict_list)
        predict_list_full.extend(predict_list)

    np.save(fpath(predict_base_dir, predict_save_file), predict_list_full)
    print('All Done, found ', len(predict_list_full), ' Matching Images. Full List Saved.')
    return predict_list_full


project = "wellcare2-"
# pretrained_model = model_used =  'vgg16'
pretrained_model = model_used = 'inceptionv3r2'
# pretrained_model = model_used =  'inceptionv3'

trial_num = 5
project_path = '/media/arun/data/wellcare2/data-aug-simple-work/'
# project_path =  '/media/arun/data/kaggle/contrast/'

# project_path =  '/media/arun/data/dogscats/'

img_height, img_width = 299, 299
epochs = 10
batch_size = 16

data_aug = True
use_learning_decay = False
# initial_learning_rate = 7.3509e-05
initial_learning_rate = 1 * 1e-4

optimizer_used = 'adam'

continue_top_training = False
load_model_archive_num = None
recompile_model = True
use_batch_norm = True

train_conv_layers = True
num_conv_layers_to_train = 10
early_stopping = False

dense_layer_neurons = 1024
dense_layer_neurons2 = 1024
dense_layer_dropout = 0.85
save_mode = 'append'

save_archive = False

# train_mode = False
# train_top   = True
# read_training_data = True
# load_top_model = False

# predict_mode = False
# move_files = False



# create the base pre-trained model
start_time = timeit.default_timer()

define_paths(project_path, project, model_used, trial_num=1)

print("Run on: ", time.asctime())
print("Project:", project, " Model: ", model_used)
print("Base Dir: ", project_path, " Model: ", model_used)
print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ", K.image_data_format())

debug_mode = False

# train_mode = 'full'
train_mode = 'bneck'


def train_model(train_mode='full'):
    if train_mode == 'full':
        model, history = train_full(continue_top_training=True, debug_mode=False)
    if train_mode == 'bneck':
        model, history = train_bottleneck_model(pre_trained_model='inceptionv3r2',
                                                load_saved_training_data=False, load_saved_model=False,
                                                train_top_using_batches=True)

    return model, history


model_history = train_model(train_mode='full')

# model, history =  train_bottleneck_model(pre_trained_model = 'inceptionv3r2', load_saved_training_data = False, load_saved_model = False)

# predict_base_dir = '/media/arun/data/dogscats/predict/'
# move_files(predict_base_dir,'ungradable')

# full_model('/media/arun/data/wellcare/non-gradable2/models/top-best-wellcare-nong-inceptionv3r2-trial-1.h5')

# predict_model = full_model('/media/arun/data/dogscats/models/top-best-wellcare2bn--inceptionv3r2-trial-1.h5')






# predict_mode(predict_model,predict_base_dir,predict_sub_folders = ['unknown'],
#             class_to_predict = 0,predict_save_file = 'predict_filenames.npy')

# print(predict_list_full[:20]) from __future__ import division
import pdb
import numpy as np
import matplotlib.pyplot as plt
import math,cv2,os,shutil
from datetime import datetime
import time,timeit
import pickle
import bcolz

from keras.models import Model,load_model
from keras.layers import Dense,Input, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras import optimizers
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras import regularizers


from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from PIL import Image
from PIL import ImageFile
# the following fixes an error in image processing - (via stackoverflow)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def fpath(dirname,filename):
    return os.path.join(dirname,filename)


def define_paths(base_dir_path,project,model_used,trial_num):
    global base_dir, train_data_dir,validation_data_dir, model_save_path
    global top_model_path, top_best_model_path, full_model_path, full_best_model_path
    global top_model_save_path

    base_dir = base_dir_path
    train_data_dir = fpath(base_dir, 'training')
    validation_data_dir = fpath(base_dir, 'validation')
    model_save_path = fpath(base_dir , 'models')
    project_fname = project + '-' + model_used + '-' + 'trial-' + str(trial_num) + '.h5'
    top_model_path  = fpath(model_save_path,'top-' +  project_fname)
    top_best_model_path  = fpath(model_save_path,  'top-best-' + project_fname)
    full_model_path  = fpath (model_save_path, 'full-' + project_fname)
    full_best_model_path  = fpath(model_save_path ,  'full-best-' + project_fname)
    top_train_features  = fpath(model_save_path, 'top_train_features,bc')
    top_validation_features  = fpath(model_save_path, 'top_validation_features,bc')
    top_model_save_path = fpath(model_save_path, 'top_model.h5')

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def to_plot(img):

    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))

def step_decay(epoch):
    initial_lrate = initial_learning_rate
    decay_rate = 0.95
    num_epochs_to_change_rate = 3.0
    # effective rate =  intial_rate* (decay_rate ^ (1+epoch/num_epochs_to_change_rate))
    lrate = initial_lrate * math.pow(decay_rate, math.floor((1+epoch)/num_epochs_to_change_rate))
    print("Learning rate set to..",lrate)
    return lrate

def find_last_conv(model,num):
    i = 0
    layer_nums = []
    for layer in model.layers:
        #print(i,layer.__class__,layer.name,"Conv" if layer.__class__.__name__ == "Conv2D" else "Non-Conv")
        if layer.__class__.__name__ == "Conv2D":
            layer_nums.append(i)
        i += 1
    # layer_nums is a list of all the Conv2D layer numbers
    # if num = 1 then layer_nums[-1] will be the number of the last conv layer
    layer_num  = layer_nums[-num]
    return layer_num




def set_conv_trainable(model,num_conv_layers_to_train):

    first_trainable_layer = find_last_conv(model,num_conv_layers_to_train)
    for layer in model.layers[:first_trainable_layer]:
       layer.trainable = False
    for layer in model.layers[first_trainable_layer:]:
       layer.trainable = True
    print("Setting conv layers from ",first_trainable_layer," as trainable ")

def compile_model(model,optimizer = 'adam'):
    # Deafaults for Adam
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0True)rad=False)
    if optimizer == 'adam':
        opt = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    if optimizer == 'sgd':
        opt = SGD(lr=initial_learning_rate, momentum=0.9)
    if optimizer == 'rmsprop':
        opt =  RMSprop(lr=initial_learning_rate, rho=0.9, epsilon=None, decay=0.0)
    print("Now compiling ",opt)
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
    print(opt," Compiled")


def  get_pretrained_model(pretrained_model):

    if pretrained_model == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
    if pretrained_model == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    if pretrained_model == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    if pretrained_model == 'inceptionv3r2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    return base_model

def run_predictions(predict_dir,predict_model):
    # run predictions on a model - returns the probabilities, and the prediction generator
    model = load_model(predict_model)
    predict_datagen = ImageDataGenerator(rescale=1. / 255)
    predict_gen =  predict_datagen.flow_from_directory(
        predict_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,shuffle=False)

    probs = model.predict_generator(predict_gen,verbose=1)
    return probs,predict_gen



def predict_images(predict_dir,predict_model,class_to_predict=0):
    # predict_dir = '/media/arun/data/wellcare/xsimple1/training'
    probs,predict_gen  = run_predictions(predict_dir,predict_model)

    where_class_true = np.argmax(probs,axis=1)==class_to_predict   #returns a boolean array where class occurs

    predict_list  = []
    for i in range(len(probs)):
        # for each index i where class is true find the corresponding filename returned in predict_gen
        if  where_class_true[i]:
            filename =  os.path.join(predict_dir,predict_gen.filenames[i])
            #display_image(filename)
            predict_list.append(filename)

    return predict_list

def display_predictions_mini(predict_list):


    fn = len(predict_list)

    cols  = 2
    rows   =  fn//cols if fn % 2  == 0 else fn//cols + 1

    #x = range(10)
    #y = range(10)

    fig, ax = plt.subplots(nrows=rows, ncols=cols,figsize=(40, 40))
    plt.subplots_adjust(wspace = 0.05,hspace = 0.70)

    i = 0
    for row in ax:
        for col in row:
            if i < fn:
                image = load_img(predict_list[i])
                image = img_to_array(image)
                col.imshow(to_plot(image))
                head,fname = os.path.split(predict_list[i])
                col.set_title(head + '\n' + fname,size=30)


            i += 1

    plt.show()

def display_predictions(predict_list):


    for img in predict_list:
        image = load_img(img)
        image = img_to_array(image)
        head,fname = os.path.split(img)
        plt.title(head + '\n' + fname)
        plt.imshow(to_plot(image))
        plt.show()



def  gen_samples(data_dir):
    print("Entering gen_samples....")
    print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ",K.image_data_format())

    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode= None,
        shuffle=False)

    print("Number of Samples = ",len(generator.filenames))
    print("Classes Found = ", generator.class_indices)
    print("Number of Classes = ",len(generator.class_indices))

    num_samples = generator.n
    num_classes = len(generator.class_indices)

    num_steps = int(math.ceil(num_samples / batch_size))
    print("Number of Training Steps (Num samples / Batch Size) =  ",num_steps)

    data_dict = dict()
    data_dict['num_samples'] = generator.n
    data_dict['num_classes'] = len(generator.class_indices)
    data_dict['class_indices'] = generator.class_indices
    data_dict['classes'] = generator.classes
    data_dict['filenames'] = generator.filenames
    data_dict['num_steps'] = num_steps


    return generator,data_dict


def  gen_train_batch(X,Y,batch_size,num_samples):
    #randomseed = 0
    #np.random.seed(randomseed)
    #np.random.shuffle(X)
    #np.random.seed(randomseed)
    #np.random.shuffle(Y)
    i = 0
    print("Generating training batch data")

    while 1:
        #randomseed += 1
        if (i + 1)*batch_size < num_samples:
            #print("\nTrain Step No. = ",i,"Y =",Y[i*batch_size:i*batch_size + 8,:] )
            yield X[i*batch_size:(i+1)*batch_size,:],Y[i*batch_size:(i+1)*batch_size,:]

            i += 1
        else:
            #print("\n Train i= ",i,"seed = ",randomseed,"i*batch_size = ",i*batch_size,"Y ",Y[i*batch_size:num_samples,:])
            #print("\nLast Train Step = ",i,"Y =",Y[i*batch_size:(i+1)*batch_size,:] )
            yield X[i*batch_size:num_samples,:],Y[i*batch_size:num_samples,:]
            i = 0
            #np.random.seed(randomseed)
            #np.random.shuffle(X)
            #np.random.seed(randomseed)
            #np.random.shuffle(Y)
            #print("\n End Train i= ",i,"seed = ",randomseed)

def  gen_val_batch(X,Y,batch_size,num_samples):
    #randomseed = 0
    #np.random.seed(randomseed)
    #np.random.shuffle(X)
    #np.random.seed(randomseed)
    #np.random.shuffle(Y)
    print("Generating validation batch data")
    i = 0

    #if debug_mode:
    #    print("Entering debug Mode...validation batch - first entry..")
    #    pdb.set_trace()

    while 1:
        #randomseed += 1

        if (i + 1)*batch_size < num_samples:

            #print("\nVal Step = ",i,"Y =",Y[i*batch_size:i*batch_size + 8,:])
            yield X[i*batch_size:(i+1)*batch_size,:],Y[i*batch_size:(i+1)*batch_size,:]
            i += 1
        else:
            #if debug_mode:
            #    print("Entering debug Mode...validation batch - last entry..")
            #    pdb.set_trace()
            #print("\nLast Val Step = ",i,"Y =",Y[i*batch_size:(i+1)*batch_size,:])
            #print("\nVal i= ",i,"seed = ",randomseed,"i*batch_size = ",i*batch_size,"Y ",Y[i*batch_size:num_samples,:])
            yield X[i*batch_size:num_samples,:],Y[i*batch_size:num_samples,:]
            i = 0
            #np.random.seed(randomseed)
            #np.random.shuffle(X)
            #np.random.seed(randomseed)
            #np.random.shuffle(Y)
            #print("\n End Valid i= ",i,"seed = ",randomseed)

def train_bottleneck_model(pre_trained_model = 'vgg16', load_saved_training_data = False, load_saved_model = False,
                           train_top_using_batches = False):
    print("Entering train_bottleneck_model..")
    print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ",K.image_data_format())

    if  load_saved_training_data:

        print("Loading training data ")
        train_data = load_array(fpath(model_save_path,'train-data.bc'))
        print("Loading training labels ")
        train_labels = load_array(fpath(model_save_path,'train-labels.bc'))
        print("Loading validation data ")
        validation_data = load_array(fpath(model_save_path,'validation-data.bc'))
        print("Loading validation labels ")
        validation_labels = load_array(fpath(model_save_path,'validation-labels.bc'))

        data_dict  = np.load(fpath(model_save_path,'train_data_dict.npy')).item()
        num_train_samples = data_dict['num_samples']
        num_steps = data_dict['num_steps']
        num_classes = data_dict['num_classes']


        data_dict  = np.load(fpath(model_save_path,'validation_data_dict.npy')).item()
        num_val_steps = data_dict['num_steps']
        num_valid_samples = data_dict['num_samples']
        if debug_mode:
                print("Loading saved data..train-data is..",train_data[:5])
                pdb.set_trace()
    else:

        print("Now  generating training data.. ")
        train_gen,data_dict = gen_samples(train_data_dir)

        num_train_samples = data_dict['num_samples']
        num_classes = data_dict['num_classes']
        num_steps = data_dict['num_steps']
        train_labels = train_gen.classes
        if debug_mode:
                print("Read training data...check the sequence  using train_gen",train_labels[:5])
                pdb.set_trace()

        train_labels = to_categorical(train_labels, num_classes=num_classes)


        np.save(fpath(model_save_path,'train_data_dict.npy'), data_dict)

        print("Generating bottleneck predictions..")
        base_model = get_pretrained_model(pretrained_model = pre_trained_model )
        train_data = base_model.predict_generator(train_gen,verbose=1)
        if debug_mode:
                print("Basemodel.predict is done..this is the train_data..",train_data[:5])
                pdb.set_trace()

        save_array(fpath(model_save_path,'train-data.bc'),train_data)
        save_array(fpath(model_save_path,'train-labels.bc'),train_labels)

        print("Generating validation data...")
        #pdb.set_trace()
        validation_gen,data_dict = gen_samples(validation_data_dir)


        num_valid_samples = data_dict['num_samples']
        num_val_steps = data_dict['num_steps']
        validation_labels = validation_gen.classes
        validation_labels = to_categorical(validation_labels, num_classes=num_classes)
        np.save(fpath(model_save_path,'validation_data_dict.npy'), data_dict)

        validation_data = base_model.predict_generator(validation_gen,verbose=1)
        save_array(fpath(model_save_path,'validation-data.bc'),validation_data)
        save_array(fpath(model_save_path,'validation-labels.bc'),validation_labels)
        if debug_mode:
                print("Basemodel.predict  done for validation data..this is the data",validation_labels[:5])
                pdb.set_trace()

    #if debug_mode:
    #    print("Entering debug Mode...training data loaded..")
    #    pdb.set_trace()

    if  load_saved_model:

        #1model = load_model(top_model_save_path)
        model = load_model(top_best_model_path)


    else:
            print("Creating the full model...")
            # now create the full model
            inputs = Input(shape = train_data.shape[1:])
            x = Flatten(input_shape=train_data.shape[1:])(inputs)
            x = GlobalAveragePooling2D()(inputs)


            x = Dense(dense_layer_neurons, activation='relu')(x)
            #x  = BatchNormalization() (x)
            x = Dropout(dense_layer_dropout) (x)

            x = Dense(dense_layer_neurons, activation='relu')(x)
            #x  = BatchNormalization() (x)
            x = Dropout(dense_layer_dropout) (x)

            x = Dense(dense_layer_neurons, activation='relu')(x)
            #x  = BatchNormalization() (x)
            x = Dropout(dense_layer_dropout) (x)

            x = Dense(dense_layer_neurons, activation='relu')(x)
            #x  = BatchNormalization() (x)
            x = Dropout(dense_layer_dropout) (x)


            predictions = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=predictions)

            #if train_conv_layers:    set_conv_trainable(model,num_conv_layers_to_train)

            compile_model(model,optimizer = optimizer_used)

            print("Model compiled...")

            # dave the top model for later
            model.save(top_model_save_path)




    checkpoint = ModelCheckpoint(filepath=top_best_model_path, monitor = 'val_acc',save_best_only=True)
    earlystop =  EarlyStopping(monitor='val_acc',  patience=3, verbose=1, mode='auto')
    lrate = LearningRateScheduler(step_decay,verbose=1)
    callback_list = [checkpoint]
    if use_learning_decay:          callback_list=callback[checkpoiint,lrate]
    if early_stopping:              callback_list =  [checkpoint,earlystop]

    #pdb.set_trace()
    print("Starting Training...")

    if train_top_using_batches:
        train_gen_bneck = gen_train_batch(train_data,train_labels,batch_size,num_train_samples)
        valid_gen_bneck = gen_val_batch(validation_data,validation_labels,batch_size,num_valid_samples)
        print("Training using batches")
        if debug_mode:
            print("Entering debug Mode...train_gen_neck")
            pdb.set_trace()

        history = model.fit_generator(
            train_gen_bneck,
            steps_per_epoch=num_steps,
            epochs=epochs,
            validation_data= valid_gen_bneck,
            validation_steps=num_val_steps, callbacks=callback_list)

        if debug_mode:
            print("Entering debug Mode...training done..")
            pdb.set_trace()

    else:

        history = model.fit(x = train_data, y = train_labels,
                        steps_per_epoch=num_steps,
                        epochs=epochs,
                        validation_data= (validation_data,validation_labels),
                        validation_steps=num_val_steps, callbacks=callback_list)



    elapsed = timeit.default_timer() - start_time

    print("Total Time Elapsed for Training = {0:.2f} ".format(elapsed))

    print("Saving Final Model ")

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    return model, history


# modes =  reg_train, top_train, predict, find_invalid   '



def load_archive():
    try:
        model_archives = np.load(os.path.join(model_save_path, 'model_archives.npy')).item()

    except FileNotFoundError:
        model_archives = dict()

    return model_archives


def move_files(predict_base_dir,folder):

    predict_list_full = np.load(fpath(predict_base_dir, 'predict_filenames.npy'))
    print("Now moving matching files...")

    dest_folder = fpath(predict_base_dir,folder)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    i = 0
    for file_name in predict_list_full:
            if not '00A-' in file_name:
                shutil.move(file_name,dest_folder)
                i += 1
    print(i, " Files Moved to",dest_folder)


def train_full(continue_top_training = False, debug_mode = False):
    print("Entering train_mode..")
    print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ",K.image_data_format())

    class_folders = os.listdir(train_data_dir)
    numclasses = len(class_folders)
    print("Found " + str(numclasses) + " classes in the Training Folder: " + train_data_dir  )

    if continue_top_training:

        model = load_model(top_best_model_path)


        if train_conv_layers:  set_conv_trainable(model,num_conv_layers_to_train)
        if recompile_model:   compile_model(model,optimizer = optimizer_used)

        if debug_mode:
            print("Entering Debug Mode... just after loading model ")
            pdb.set_trace()


    else:
        base_model =  get_pretrained_model(pretrained_model)
        if debug_mode:
            print("Entering debug Mode...just after compile model")
            pdb.set_trace()

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(dense_layer_neurons, activation='relu')(x)
        if use_batch_norm: x  = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout) (x)
        x = Dense(dense_layer_neurons, activation='relu')(x)
        if use_batch_norm:  x = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout) (x)
        x = Dense(dense_layer_neurons, activation='relu')(x)
        #x  = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout) (x)
        x = Dense(dense_layer_neurons, activation='relu')(x)
        #if use_batch_norm:  x = BatchNormalization() (x)
        x = Dropout(dense_layer_dropout) (x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(numclasses, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        if train_conv_layers:    set_conv_trainable(model,num_conv_layers_to_train)

        compile_model(model,optimizer = optimizer_used)

        print("Setting Up Image Data Generator...")
        if debug_mode:
            print("Entering debug Mode...just after compile model")
            pdb.set_trace()

    #bottleneck_features_train = base_model.predict_generator(train_generator, steps = num_steps,verbose = 1)


    if data_aug:

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.05,
            rotation_range=25,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            channel_shift_range = 0.15,
            horizontal_flip=True,
            fill_mode = 'constant')
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)






    print("Setting up Training Generator")
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    nb_train_samples = len(train_generator.filenames)
    num_classes = len(train_generator.class_indices)

    print("Num Training Examples =  ",nb_train_samples)




    print("Setting up Validation Generator")

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    nb_validation_samples = len(validation_generator.filenames)
    print("Num Validation  Samples ",nb_validation_samples)



    classwts = {0: 4.884889059271241,
     1: 1.0,
     2: 5.860233990285925,
     3: 8.952072516664096,
     4: 6.659112346088153,
     5: 10.514077562438366,
     6: 10.726716491458607,
     7: 10.840229074158607,
     8: 10.804359515241758,
     9: 10.893346400050042}


    checkpoint = ModelCheckpoint(filepath=top_best_model_path, monitor = 'val_acc',save_best_only=True)
    earlystop =  EarlyStopping(monitor='val_loss',  patience=5, verbose=1, mode='auto')
    lrate = LearningRateScheduler(step_decay,verbose=1)
    #callback_list = [checkpoint,lrate]
    if use_learning_decay:
        callback_list=[checkpoint,lrate]
    else:
        callback_list=[checkpoint]

    #if early_stopping:              callback_list = callback_list.append(earlystop)

    print("Starting Training...")
    if debug_mode:
        print("Entering debug Mode...just before Model.fit_generator")
        pdb.set_trace()

    history1 = model.fit_generator(
        train_generator,
        steps_per_epoch=int(math.ceil(nb_train_samples / batch_size)),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=int(math.ceil(nb_validation_samples / batch_size)),callbacks=callback_list)


    #class_weight = classwts,

    elapsed = timeit.default_timer() - start_time

    print("Total Time Elapsed for Training = {0:.2f} ".format(elapsed))
    print("Best Model Saved as ", top_best_model_path)


    print("Saving Final Model ")
    model.save(top_model_path)  # creates a HDF5 file 'my_model.h5'

    print("Final Model Saved as ", top_model_path)

    finish_time = time.asctime()
    print("All done at: ",finish_time)

    if save_archive:


        print(" Now saving model archive file...")
        model_archive = dict()
        model_archive["project"] = project
        model_archive["model_used"] = model_used
        model_archive["base_dir"]  = base_dir
        model_archive["trial_num"] = trial_num
        model_archive["numclasses"]  = numclasses
        model_archive["epochs"]  = epochs
        model_archive["batch_size"]  = batch_size
        model_archive["initial_learning_rate"] = initial_learning_rate
        model_archive["image_size"] =  (img_height, img_width )
        model_archive["data_aug"]  = data_aug
        model_archive["nb_training_samples"]  = nb_train_samples
        model_archive["nb_validation_samples"]  = nb_validation_samples
        model_archive["Elapsed_1"]  = elapsed
        model_archive["history1"]  = history1
        model_archive["model_paths"]  = {"top_best_model_path" : top_best_model_path,
                                         "top_model_path": top_model_path }
        model_archive["continue_top_training"] = continue_top_training
        model_archive["use_learning_decay"]  = use_learning_decay
        model_archive["optimizer_used"]  = optimizer_used
        model_archive["recompile_model"] = recompile_model
        model_archive["train_conv_layers"] = train_conv_layers
        model_archive["num_conv_layers_to_train"] = num_conv_layers_to_train
        model_archive["early_stopping"] = early_stopping
        model_archive["dense_layer_neurons"] = dense_layer_neurons
        model_archive["dense_layer_dropout"] = dense_layer_dropout
        model_archive["validation_accuracy"] = history1.history["val_acc"]
        model_archive["training_accuracy"] = history1.history["acc"]

        model_archive["start_time"] = start_time
        model_archive["finish_time"] =  finish_time



        model_archives = load_archive()
        t = time.asctime()
        model_archives[t] = model_archive


        #if debug_mode:
        #print("Entering debug Mode...just before returning from function")
        #pdb.set_trace()

        np.save(os.path.join(model_save_path, 'model_archives.npy'),model_archives)
        print("Model archive saved..all done!")

    if debug_mode:
        print("Entering debug Mode...just before returning from function")
        pdb.set_trace()

    return model,history1

def full_model(model_path):
    # this combines the pre-trained base model and its
    # predictions with  the top model to create the full
    # model that can then be used for predictions

    base_model =  get_pretrained_model(pretrained_model)
    print("Base Model loaded..now loading top model..")
    top_model = load_model(model_path)

    print("Top model loaded...")
    if K.image_data_format() == "channels_first":
        inputshape = (3,img_height, img_width)
    else:
        inputshape = (img_height, img_width,3)
    main_input  = Input(shape = inputshape)
    main_output  = base_model(main_input)

    top_output = top_model(main_output)

    full_model = Model(inputs = main_input, outputs = top_output)

    compile_model(full_model,optimizer = optimizer_used)

    full_model.save(fpath(model_save_path, 'full-model.h5'))
    #return full_model
    print("Full Model Saved")
    return fpath(model_save_path, 'full-model.h5')


def predict_mode(predict_model,predict_base_dir,predict_sub_folders = ['test','training','validation'],
                 class_to_predict = 0,predict_save_file = 'predict_filenames.npy',disp_pred = False ):

    predict_dir_list = []


    #for  f in ['test','training','validation']:
    for  f in predict_sub_folders:
        predict_dir_list.append(fpath(predict_base_dir,f))

    predict_list_full = []
    for predict_dir in predict_dir_list:
        predict_list = predict_images(predict_dir,predict_model,class_to_predict=class_to_predict)
        print("Found ",len(predict_list), " Matching Images in...",predict_dir)
        if disp_pred:  display_predictions(predict_list)
        predict_list_full.extend(predict_list)

    np.save(fpath(predict_base_dir,predict_save_file),predict_list_full)
    print('All Done, found ', len(predict_list_full),' Matching Images. Full List Saved.' )
    return predict_list_full






project = "wellcare2-"
#pretrained_model = model_used =  'vgg16'
pretrained_model = model_used =  'inceptionv3r2'
#pretrained_model = model_used =  'inceptionv3'

trial_num  = 5
project_path =  '/media/arun/data/wellcare2/data-aug-simple-work/'
#project_path =  '/media/arun/data/kaggle/contrast/'

#project_path =  '/media/arun/data/dogscats/'

img_height, img_width = 299,299
epochs =  10
batch_size = 16

data_aug = True
use_learning_decay = False
#initial_learning_rate = 7.3509e-05
initial_learning_rate = 1*1e-4


optimizer_used  = 'adam'

continue_top_training = False
load_model_archive_num = None
recompile_model = True
use_batch_norm = True

train_conv_layers = True
num_conv_layers_to_train = 10
early_stopping = False

dense_layer_neurons = 1024
dense_layer_neurons2 = 1024
dense_layer_dropout = 0.85
save_mode = 'append'

save_archive = False

#train_mode = False
#train_top   = True
#read_training_data = True
#load_top_model = False

#predict_mode = False
#move_files = False



# create the base pre-trained model
start_time = timeit.default_timer()


define_paths(project_path,project, model_used,  trial_num=1)

print("Run on: ", time.asctime())
print("Project:",project," Model: ",model_used)
print("Base Dir: ", project_path," Model: ",model_used)
print("Image Dim Ordering = ", K.image_dim_ordering(), " Image Data Format = ",K.image_data_format())



debug_mode = False

#train_mode = 'full'
train_mode = 'bneck'

def train_model(train_mode = 'full'):

    if train_mode == 'full':
        model,history = train_full(continue_top_training = True, debug_mode = False)
    if train_mode == 'bneck':
        model,history =  train_bottleneck_model(pre_trained_model = 'inceptionv3r2',
                        load_saved_training_data = False, load_saved_model = False,
                        train_top_using_batches = True)

    return model,history

model_history = train_model( train_mode = 'full')

#model, history =  train_bottleneck_model(pre_trained_model = 'inceptionv3r2', load_saved_training_data = False, load_saved_model = False)

#predict_base_dir = '/media/arun/data/dogscats/predict/'
#move_files(predict_base_dir,'ungradable')

#full_model('/media/arun/data/wellcare/non-gradable2/models/top-best-wellcare-nong-inceptionv3r2-trial-1.h5')

#predict_model = full_model('/media/arun/data/dogscats/models/top-best-wellcare2bn--inceptionv3r2-trial-1.h5')






#predict_mode(predict_model,predict_base_dir,predict_sub_folders = ['unknown'],
#             class_to_predict = 0,predict_save_file = 'predict_filenames.npy')

#print(predict_list_full[:20])