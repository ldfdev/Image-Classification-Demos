import PIL
import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt


if __name__=='__main__':
    imagePath = '/home/loxor/Documents/Humpback_Whale_Identification/data/some_imgs_from_train/'
    image = PIL.Image.open(os.path.join(imagePath, '0c5537f22.jpg'))
    image = TF.crop(image, 200, 200, 1000, 1000)
    image = TF.resize(image, (2000,2000))
    plt.imshow(image)
    plt.show()