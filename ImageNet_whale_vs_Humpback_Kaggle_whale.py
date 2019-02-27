import numpy as np
import matplotlib.pyplot as plt
import PIL

imageNetPath = '../data/ImageNet_humpback_dataset/data/'
imageNet_images = ['_39052101_humpback_203_noaa.jpg',
                   '08-08p10breaching.jpg',
                   '00176.jpg',
                   '03892.jpg'
                  ]

humpbackImagesPath = '../data/some_imgs_from_train/'
humpback_images = [
                    '0ad6ea37e.jpg',
                    '0b2565d62.jpg',
                    '0c3197e04.jpg',
                    '0af805558.jpg'
                  ]

image_size = (300,300)



def create_hstack_of_images(imagesPath, images_list):
    num = len(images_list)
    for i in range(num):
        new_image = PIL.Image.open(imagesPath + images_list[i]).resize(image_size)
        if i == 0:
            image_concat = new_image
        else:
            image_concat = np.hstack((image_concat, new_image))
    return image_concat

imageNet_hstack = create_hstack_of_images(imageNetPath, imageNet_images)
humpback_hstack = create_hstack_of_images(humpbackImagesPath, humpback_images)
print('imageNet {} humpback {}'.format(imageNet_hstack.shape, humpback_hstack.shape))
plt.imshow(np.vstack((imageNet_hstack, humpback_hstack)))
plt.show()