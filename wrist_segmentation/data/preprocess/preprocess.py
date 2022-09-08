'''
There are functions to load and transform the data to required form.
Initially, we had datasets in form of .mat files, but with different structure.

If new data will be added you should add your function here and use it later.
'''


import numpy as np
import cv2


def rotate(image):
    if np.size(image, 1) > np.size(image, 0):
        image = np.transpose(image)
    return image


def make_crops(arr, size):
    '''
    Make all possible crops from image
    :param arr: 2dim array for cropping
    :param size: size of crops
    :return: list of all possible crops
    '''
    assert len(arr.shape) == 2, 'arr size must be (x,y)'
    assert arr.shape[0] >= size[0] and arr.shape[1] >= size[1], 'crop size > arr size'
    crops = []
    y_num, x_num = arr.shape[0] // size[0] + 1, arr.shape[1] // size[1] + 1
    for i in range(x_num):
        for j in range(y_num):
            l_x = i * (arr.shape[0] - size[0])
            l_y = j * (arr.shape[1] - size[1])
            crops.append(arr[l_x:l_x + size[0], l_y:l_y + size[1]])
    return crops


def center_crop(image: np.ndarray, crop_size: np.ndarray):
    '''
    Make center crop of image using only sizes
    :param image:
    :param crop_size: size of the crop
    :return: cropped image,
    '''
    image_size = np.array(image.shape)
    crop_size = np.array(crop_size)
    assert len(image_size) == 2 and len(crop_size) == 2, 'Only 2 dims accepted'
    assert image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1], 'Crop size smaller than image size'

    left = np.array([0, 0])
    index = image_size > crop_size

    left[index] = (image_size[index] - crop_size[index]) / 2
    right = image_size - left

    return image[left[0]:right[0], left[1]:right[1]]


def expand_img(image, size, with_noise=False):
    '''
    Expand image size to size with noise or zeros
    :param image: image to expand
    :param size: expand image to this size
    :param with_noise: expand with noise or zeros
    :return: expanded to the size image
    '''
    assert np.size(image, 0)*np.size(image, 1) < size[0]*size[0], 'The image size larger the expanded size'

    if with_noise:
        expanded = np.zeros(size)
    else:
        expanded = np.sqrt(np.random.randn(*size) ** 2 + np.random.randn(*size) ** 2)
    expanded[:np.size(image, 0), :np.size(image, 1)] = image[:size[0], :size[1]]
    return expanded


def resize_img(image, size):
    '''
    Just wrapper of cv2.resize
    :param image:
    :param size:
    :return: resized image
    '''
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def white_norm_img(image):
    return (image - np.mean(image)) / (np.std(image))


def linear_norm_img(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def full_preprocess(image, mask, size, resize_thresh = 1.7):
    im = rotate(image)
    ma = rotate(mask)
    if (np.size(im, 0) >= resize_thresh * size[0] and
            np.size(im, 1) >= resize_thresh * size[1]):
        im = resize_img(im,size)
        ma = resize_img(ma, size)
    elif np.size(im, 0) < size[0] or np.size(im, 1) < size[1]:
        im = expand_img(im,size)
        ma = expand_img(ma, size)
    else:
        im = center_crop(im, size)
        ma = center_crop(ma, size)

    im = white_norm_img(im)
    return im, ma


def ft_preprocess(image, mask, size, resize_thresh = 1.7):
    im = rotate(image)
    ma = rotate(mask)
    if (np.size(im, 0) >= resize_thresh * size[0] and
            np.size(im, 1) >= resize_thresh * size[1]):
        im = resize_img(im,size)
        ma = resize_img(ma, size)
    elif np.size(im, 0) < size[0] or np.size(im, 1) < size[1]:
        im = expand_img(im,size)
        ma = expand_img(ma, size)
    else:
        # find crop with maximum amount of cartilage
        crops = make_crops(ma, size) # find all crops
        sums = [np.sum(crop) for crop in crops] # amount of cartilage
        argmax = np.argmax(sums) # find maximum

        ma = crops[argmax]
        crops = make_crops(im, size)
        im = crops[argmax]

    im = white_norm_img(im)
    return im, ma


def crop_preprocess(image, mask, size):
    im = rotate(image)
    im = center_crop(im, size)
    im = white_norm_img(im)

    ma = rotate(mask)
    ma = center_crop(ma, size)
    return im, ma


def cv_preprocess(image, mask):
    return white_norm_img(image), mask


def knee_preprocess(inputs):
    outputs = inputs
    data = inputs['data']

    for key in data.keys():
        if 'imag' in key:
            data[key] = white_norm_img(data[key])
        data[key] = np.swapaxes(data[key],0,-1)

    outputs['data'] = data

    return outputs


def preprocess_wrapper(inputs):
    '''
    Wrapper to use in loader in order to follow the same structure of functions.
    You should pass to this function dict with the required args
    :param inputs: dict of the arguments for the preprocessing function
    :return: image, mask or only image or something else, depending on preprocess_func
    '''
    assert 'preprocess_func' in inputs.keys(), 'There is no preprocess_func in args dict'
    req_args = inputs['preprocess_func'].__code__.co_varnames
    pass_args = {}
    for arg in req_args:
        if arg in inputs.keys():
            pass_args[arg] = inputs[arg]

    return inputs['preprocess_func'](**pass_args)
