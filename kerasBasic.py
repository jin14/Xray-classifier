import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold, KFold
from keras.layers import Input, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import vgg16, resnet50
from keras import optimizers

def createVGG(dim, weights, lr, train_last_layer=False):
    input = Input(shape=(dim, dim, 3), name="image_input")
    vgg = vgg16.VGG16(include_top=False, weights=weights)

    if train_last_layer:
        set_trainable = False
        for layers in vgg.layers:
            if layers.name == "block5_conv1":
                set_trainable = True
            if set_trainable:
                layers.trainable = True
            else:
                layers.trainable = False

    output_vgg = vgg(input)

    out = Flatten(name='flatten')(output_vgg)
    out = Dense(2048, activation='relu', name='fc1')(out)
    out = Dropout(0.7)(out)
    out = Dense(2048, activation='relu', name='fc2')(out)
    out = Dropout(0.7)(out)
    out = Dense(15, activation='softmax', name='predictions')(out)

    vgg_modified = Model(input=input, output=out)
    vgg_modified.compile(optimizer=optimizers.RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return vgg_modified


def generatePlot(acc, val_acc, loss, val_loss, dir, name, lr):

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'b*-', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'r*-', label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(dir + name + '_acc_{}.png'.format(lr))

    plt.figure()
    plt.plot(epochs, loss, 'bo-', label="Training Loss")
    plt.plot(epochs, val_loss, 'ro-', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dir + name + '_loss_{}.png'.format(lr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply image directory and label filename")
    parser.add_argument('--t', help="image directory")
    parser.add_argument('--lt', help="label file name")
    parser.add_argument('--p', help="directory to save plot")
    parser.add_argument('--b', type=int, help="batch size")
    parser.add_argument('--w', help="imagenet:pretrained weights from keras or None:random initialization")
    parser.add_argument('--d', type=int, help="image width or length, both should be the same")
    parser.add_argument('--k', type=int, help="K fold cross validation")
    parser.add_argument('--e', type=int, help="No of epochs")
    parser.add_argument('--tr',type=int, help="Is last layer trainable: 0 - false, 1 True")

    args = parser.parse_args()

    BATCH_SIZE = args.b
    WEIGHTS = args.w if args.w == 'imagenet' else None
    DIM = args.d
    EPOCHS = args.e
    # learning_rates = [0.001, 0.01, 0.0001, 0.00001]
    learning_rates = [0.0001]

    df = pd.read_csv(args.t + args.lt)
    img_indexes = list(df.index)
    kfold = KFold(n_splits=args.k, shuffle=True)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )
    valid_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    for lr in learning_rates:
        print("Learning rate:", lr)
        model = createVGG(DIM, WEIGHTS, lr, bool(args.tr))
        model.summary()

        acc = []
        val_acc = []
        loss = []
        val_loss = []

        for train_index, test_index in kfold.split(img_indexes):
            traindf = df.iloc[train_index, :].reset_index()
            validdf = df.iloc[test_index, :].reset_index()

            train_generator = train_datagen.flow_from_dataframe(dataframe=traindf,
                                                                directory=args.t,
                                                                x_col="image",
                                                                y_col="label",
                                                                has_ext=True,
                                                                class_mode="categorical",
                                                                target_size=(DIM, DIM),
                                                                batch_size=BATCH_SIZE)

            valid_generator = valid_datagen.flow_from_dataframe(dataframe=validdf,
                                                                directory=args.t,
                                                                x_col="image",
                                                                y_col="label",
                                                                has_ext=True,
                                                                class_mode="categorical",
                                                                target_size=(DIM, DIM),
                                                                batch_size=BATCH_SIZE)

            result = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=len(traindf.index) / BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=valid_generator,
                validation_steps=len(validdf.index) / BATCH_SIZE,
                workers=32,
                use_multiprocessing=True
            )

            model.save('model_edge_{}_{}_{}'.format(lr, BATCH_SIZE, DIM))

            acc.extend(result.history['acc'])
            val_acc.extend(result.history['val_acc'])
            loss.extend(result.history['loss'])
            val_loss.extend(result.history['val_loss'])


        generatePlot(acc, val_acc, loss, val_loss, args.p, 'vgg16_' + str(BATCH_SIZE) + '_' + str(DIM) + '_' + str(time.time()), str(lr))