import pandas as pd
import random
import cv2
import argparse
import os
from multiprocessing import Pool
from itertools import repeat
"""
Returns a dictionary where the keys are the disease label and the values are a list of image file name

:params filepath: filepath of the csv file
:type filepath: str
"""


def breakdown(filepath, single=True):
    data = pd.read_csv(filepath)
    images = data[['Image Index','Finding Labels']]
    if single:
        images['single'] = images['Finding Labels'].apply(lambda x: 1 if len(x.split('|')) == 1 else 0)
        images_split = images[images['single']==1][['Finding Labels','Image Index']]
        images_split.columns = ['Label', 'Image']
    else:
        images_split = pd.DataFrame(images['Finding Labels'].str.split('|').tolist(), index=images['Image Index']).stack()
        images_split = images_split.reset_index()[[0, 'Image Index']]
        images_split.columns = ['Label', 'Image']

    image_dict = images_split.set_index('Label').stack().groupby(level=0).apply(list).to_dict()

    return image_dict


""" 
Return all filenames of images that will make up the training data.
It ensures each disease class has the same number of images

:param img_dict: name of variable containing the dictonary of disease labels and corresponding image name
:type img_dict: dict
:param n: number of image each class will have, ideally should be less than class with the minimum number of image. In the single case its hernia 110
:type n: int
:param seed: seed for randomization
:type seed: int 
"""


def uniformDist(img_dict, n, seed):
    training_data = []
    mapping_labels = {v:i for i,v in enumerate(sorted(img_dict.keys()))}
    overview = {'image':[], 'label':[]}
    random.seed(seed)
    for key,value in img_dict.items():
        random.shuffle(value)
        if n < len(value):
            selected = value[:n]
            training_data.extend(selected)
            overview['image'].extend(selected)
            overview['label'].extend([mapping_labels[key]]*n)
        else:
            print("N:{} is bigger than number of images available {}".format(n, len(value)))
            selected = value[:]
            training_data.extend(selected)
            overview['image'].extend(selected)
            overview['label'].extend([mapping_labels[key]]*len(value))

    return training_data, overview, mapping_labels


""" 
Return all filenames of images that will make up the training data.
It ensures each disease class has the same number of images

:param img_dict: name of variable containing the dictonary of disease labels and corresponding image name
:type img_dict: dict
:param n: number of image each class will have, ideally should be less than class with the minimum number of image. In the single case its hernia 110
:type n: int
:param seed: seed for randomization
:type seed: int 
"""


def preprocessAll(img_dict,seed):
    training_data = []
    mapping_labels = {v:i for i,v in enumerate(sorted(img_dict.keys()))}
    overview = {'image':[], 'label':[]}
    random.seed(seed)
    for key,value in img_dict.items():
        random.shuffle(value)
        training_data.extend(value)
        overview['image'].extend(value)
        overview['label'].extend([mapping_labels[key]]*len(value))

    return training_data, overview, mapping_labels


"""
Return a resized image

:param img: file path of the image
:type img: str
:param x: width - no of pixels
:type x: int
:param y:height - no of pixels
:type y: int

"""


def resize(img,x,y):
    return cv2.resize(img, (x,y), interpolation=cv2.INTER_AREA)


"""
Return a histogram equalized image

:param img: image to be equalized 
:type img: str
"""


def equilize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


"""
Return a adaptive histogram equalized image

:param img: image to be equalized 
:type img: str
"""


def adaptiveEqualize(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    img_output = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return img_output

def process(image, image_dir, out_dir):
    image_path = image_dir + image
    img = cv2.imread(image_path)
    img = resize(img, 128, 128)
    # equalize hist
    # img = equilize(img)
    # adaptive equalize
    img = adaptiveEqualize(img)
    cv2.imwrite(out_dir + image, img)
    print("Image {} processed successfully!".format(image))


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Process filepaths")
    parser.add_argument('--c', help="File path of data entry file")
    parser.add_argument('--i', help="File path of original image directory")
    parser.add_argument('--o', help="File path of processed image directory")
    args = parser.parse_args()
    file = args.c
    image_dir = args.i
    out_dir = args.o
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    distribution = breakdown(file,False)
    sampled_images, sampled_overview, mappings = preprocessAll(distribution,42)
    sampled_df = pd.DataFrame(data=sampled_overview)
    sampled_df.to_csv(out_dir + "image_labels.csv", index=False)
    mappings_df = pd.DataFrame(list(mappings.items()), columns=['Disease','Label'])
    mappings_df.to_csv(out_dir + "mappings.csv", index=False)
    print("Size of training data: " + str(len(sampled_images)), "\n Number of unique data: " + str(len(set(sampled_images))))

    #sampled_images = [i for i in sampled_images if int(i.split('_')[0].lstrip("0")) < 100]

    with Pool() as pool:
        pool.starmap(process, zip(sampled_images, repeat(image_dir), repeat(out_dir)))

    print("All images processed successfully!")
    # for image in sampled_images:
    #     print("Processing image:" + image)
    #     image_path = image_dir + image
    #     img = cv2.imread(image_path)
    #     img = resize(img, 128, 128)
    #     # equalize hist
    #     # img = equilize(img)
    #     # adaptive equalize
    #     img = adaptiveEqualize(img)
    #     cv2.imwrite(out_dir + image, img)
    #     print("Image {} processed successfully!".format(image))
    # print("All images processed successfully!")
