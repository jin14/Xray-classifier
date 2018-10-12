import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from keras.layers import Input, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import vgg16, resnet50
from keras import optimizers
# tensor-gpu 1.9.0

def createVGG(dim, weights, lr):

    input = Input(shape=(dim, dim, 3), name="image_input")
    vgg = vgg16.VGG16(include_top=False, weights=weights)
    output_vgg = vgg(input)

    out = Flatten(name='flatten')(output_vgg)
    out = Dense(4096, activation='relu', name='fc1')(out)
    out = Dense(4096, activation='relu', name='fc2')(out)
    out = Dense(15, activation='softmax', name='predictions')(out)

    vgg_modified = Model(input=input, output=out)
    vgg_modified.compile(optimizer=optimizers.RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return vgg_modified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply image directory and label filename")
    parser.add_argument('--t', help="image directory")
    parser.add_argument('--lt', help="label file name")
    parser.add_argument('--b', type=int, help="batch size")
    parser.add_argument('--w', help="imagenet:pretrained weights from keras or none:random initialization")
    parser.add_argument('--d', type=int, help="image width or length, both should be the same")
    parser.add_argument('--k', type=int, help="K fold cross validation")
    parser.add_argument('--e', type=int, help="No of epochs")

    args = parser.parse_args()

    BATCH_SIZE = args.b
    WEIGHTS = args.w
    DIM = args.d
    EPOCHS = args.e
    learning_rates = [0.001, 0.01, 0.0001, 0.00001]

    df = pd.read_csv(args.t + args.lt)
    img_indexes = list(df.index)
    kfold = KFold(n_splits=args.k, shuffle=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255)
    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )

    for lr in learning_rates:
        print("Learning rate:", lr)
        model = createVGG(args.d, args.w, lr)
        print("Model Summary: ", model.summary())

        for train_index, test_index in kfold.split(img_indexes):
            traindf = df.iloc[train_index, :].reset_index()
            validdf = df.iloc[test_index, :].reset_index()

            train_generator = train_datagen.flow_from_dataframe(dataframe=traindf,
                                                                directory=args.t,
                                                                x_col="image",
                                                                y_col="label",
                                                                has_ext=True,
                                                                class_mode="categorical",
                                                                target_size=(DIM,DIM),
                                                                batch_size=BATCH_SIZE)

            valid_generator = valid_datagen.flow_from_dataframe(dataframe=validdf,
                                                                directory=args.t,
                                                                x_col="image",
                                                                y_col="label",
                                                                has_ext=True,
                                                                class_mode="categorical",
                                                                target_size=(DIM,DIM),
                                                                batch_size=BATCH_SIZE)

            model.fit_generator(
                generator=train_generator,
                steps_per_epoch=len(traindf.index)/BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=valid_generator,
                validation_steps=len(validdf.index)/BATCH_SIZE
            )
