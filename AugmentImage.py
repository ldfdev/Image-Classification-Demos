import torchvision, torch
from copy import deepcopy
import numpy as np

class ImageTransformations():
    __all__ = ['get_list_of_transformations', '__call__']
    '''
    All images taken as parameters by class methods must be PIL format

    Example usage:

    transforms, decriptive_transformations = ImageTransformations()(img)
        where listImgs is a list of 26 image transformations
              decriptive_transformations is a list of string containing a short description of each transformatiion
    '''
    standard_dimension = 299
    @staticmethod
    def resize_if_needed(image):
        width, height = image.size
        if (width < ImageTransformations.standard_dimension):
            image_transform = torchvision.transforms.Resize((ImageTransformations.standard_dimension,height))
        if (height < ImageTransformations.standard_dimension):
            image_transform = torchvision.transforms.Resize((width, ImageTransformations.standard_dimension))
        return image_transform
    @staticmethod
    def augment_TenCrop(img):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(ImageTransformations.resize_if_needed),
            torchvision.transforms.TenCrop((ImageTransformations.standard_dimension,ImageTransformations.standard_dimension))]) # FiveCrop + HorizontalFlip
        return list(transform(img)) # by default TenCrop return a tuple
    
    @staticmethod
    def general_transform():
        return torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=.04, hue=.05, saturation=.05),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.Resize((ImageTransformations.standard_dimension,ImageTransformations.standard_dimension))
            ])
    
    @staticmethod
    def gaussian_noise(img, mean=0, stddev=0.1):
        noise = torch.empty(img.size())
        torch.nn.init.normal_(noise, mean, stddev)
        return img + noise
    @staticmethod
    def add_noise_and_saturate(img):
        return torchvision.transforms.Compose([
            ## saturation in clipping values outside of [0.0, 1.0] to 0 or 1
            torchvision.transforms.Lambda(ImageTransformations.gaussian_noise),
            torchvision.transforms.Lambda(lambda x: np.clip(x, 0, 1))])(img)
    @staticmethod
    def get_list_of_transformations(img):
        img = ImageTransformations.resize_if_needed
        l = [ImageTransformations.general_transform,
             torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), \
                                             torchvision.transforms.Resize((ImageTransformations.standard_dimension,ImageTransformations.standard_dimension))]),
             torchvision.transforms.Compose([torchvision.transforms.RandomRotation(17),\
                                             torchvision.transforms.Resize((ImageTransformations.standard_dimension,ImageTransformations.standard_dimension))])] \
             + ImageTransformations.augment_TenCrop
        l = list(map(lambda x:  torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.443,0.453,0.461], [0.51,0.48,0.5])
            ]), l))
        transformations = [
            'brightness_flip_rotation',
            'hflip',
            'rotation',
            'crop_tl',
            'crop_tr',
            'crop_bl',
            'crop_br',
            'crop_center',
            'flipped_crop_tl',
            'flipped_crop_tr',
            'flipped_crop_bl',
            'flipped_crop_br',      
            'flipped_crop_center',
            'noise_brightness_flip_rotation',
            'noise_hflip',
            'noise_rotation',
            'noise_crop_tl',
            'noise_crop_tr',
            'noise_crop_bl',
            'noise_crop_br',
            'noise_crop_center',
            'noise_flipped_crop_tl',
            'noise_flipped_crop_tr',
            'noise_flipped_crop_bl',
            'noise_flipped_crop_br',
            'noise_flipped_crop_center'
        ]
        return (l + [ImageTransformations.add_noise_and_saturate(i) for i in l], transformations)
    def __call__(self, img):
        return self.get_list_of_transformations(img)

class TransformedImages():
    __all__ = ['get_list_of_transformations', '__call__']
    '''
    All images taken as parameters by class methods must be PIL format

    Example usage:

    listImgs, decriptive_transformations = TransformedImages()(img)
        where listImgs is a list of 26 images obtained from img, by applying transformations
              decriptive_transformations is a list of string containing a short description of each transformatiion
    '''
    standard_dimension = 240
    @staticmethod
    def resize_if_needed(image):
        width, height = image.size
        if (width < TransformedImages.standard_dimension):
            image=torchvision.transforms.Resize((TransformedImages.standard_dimension,height))(image)
        if (height < TransformedImages.standard_dimension):
            image=torchvision.transforms.Resize((width, TransformedImages.standard_dimension))(image)
        return image
    @staticmethod
    def augment_FiveCrop(image):
        crop_height, crop_width = [x // 2 for x in image.size]
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(TransformedImages.resize_if_needed),
            torchvision.transforms.FiveCrop((crop_width, crop_height))]) # 4 corners + centercrop
        return list(transform(image)) # by default FiveCrop return a tuple
    
    @staticmethod
    def add_noise_and_saturate(image):
        def gaussian_noise(image, mean=0, stddev=0.1):
            noise = torch.empty(image.size())
            torch.nn.init.normal_(noise, mean, stddev)
            return image + noise
        
        return torchvision.transforms.Compose \
            ([
            # saturation in clipping values outside of [0.0, 1.0] to 0 or 1
             torchvision.transforms.Lambda(gaussian_noise),
             torchvision.transforms.Lambda(lambda x: np.clip(x, a_min=-1, a_max=1))
            ])(image)
    @staticmethod
    def get_list_of_transformations(image):
        image = TransformedImages.resize_if_needed(image)
        l = [
              image,
              torchvision.transforms.ColorJitter(brightness=.04, hue=.05, saturation=.05)(image),
              torchvision.transforms.RandomAffine(degrees=(-20,20),
                                                  translate=(0.3,0.3),
                                                  shear=(-10,10))(image),
              torchvision.transforms.RandomRotation((-20,20))(image)
            ] + TransformedImages.augment_FiveCrop(image)
        l = list(map(lambda image:  torchvision.transforms.Compose \
            ([
              torchvision.transforms.Resize((TransformedImages.standard_dimension,TransformedImages.standard_dimension)),
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])(image), l))
        transformations = [
            ''
        ]
        return (l + [TransformedImages.add_noise_and_saturate(image) for image in l], transformations)
    def __call__(self, image):
        return self.get_list_of_transformations(image)

class Patches_of_Images(object):
    __all__ = ['get_list_of_transformations', '__call__']
    '''
    All images taken as parameters by class methods must be PIL format

    Example usage:

    listImgs, decriptive_transformations = Patches_of_Images()(img)
        where listImgs is a list of images obtained from img, by extracting patches
              decriptive_transformations is a list of string containing a short description of each transformation
    '''
    standard_dimension = 240
    @staticmethod
    def resize_if_needed(image):
        height, width = image.size
        if (width < TransformedImages.standard_dimension):
            image=torchvision.transforms.Resize((TransformedImages.standard_dimension,height))(image)
        if (height < TransformedImages.standard_dimension):
            image=torchvision.transforms.Resize((width, TransformedImages.standard_dimension))(image)
        return image
    @staticmethod
    def augment_Patches(image):
        stride = 90
        image = torchvision.transforms.Lambda(TransformedImages.resize_if_needed)(image)
        image_height, image_width = image.size
        crop_height, crop_width = [(3 * x) // 4 for x in image.size]
        # print('Image dimensions {image_height}x{image_width}, crop dimensions {crop_height}x{crop_width}'.format\
        #         (image_height=image_height,
        #          image_width=image_width,
        #          crop_height=crop_height,
        #          crop_width=crop_width))
        patches = [image]
        top_h, top_w = 0,0
        while((top_h + crop_height <= image_height) and \
              (top_w + crop_width <= image_width)):
            patches.append(
                torchvision.transforms.functional.crop(image, top_w, top_h, crop_width, crop_height))
            # print("patch dims", top_h + crop_height, top_w + crop_width)
            top_w += stride
            if (top_w + crop_width > image_width):
                top_w, top_h = 0, top_h + stride
        print('patches created:', len(patches))
        return patches
    
    @staticmethod
    def add_noise_and_saturate(image):
        def gaussian_noise(image, mean=0, stddev=0.1):
            noise = torch.empty(image.size())
            torch.nn.init.normal_(noise, mean, stddev)
            return image + noise
        
        return torchvision.transforms.Compose \
            ([
            # saturation in clipping values outside of [0.0, 1.0] to 0 or 1
             torchvision.transforms.Lambda(gaussian_noise),
             torchvision.transforms.Lambda(lambda x: np.clip(x, a_min=-1, a_max=1))
            ])(image)
    @staticmethod
    def get_list_of_transformations(image):
        image = TransformedImages.resize_if_needed(image)
        l = Patches_of_Images.augment_Patches(image)
        l = list(map(lambda image:  torchvision.transforms.Compose \
            ([
              torchvision.transforms.Resize((TransformedImages.standard_dimension,TransformedImages.standard_dimension)),
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])(image), l))
        transformations = [
            ''
        ]
        return (l + [TransformedImages.add_noise_and_saturate(image) for image in l], transformations)
    def __call__(self, image):
        return self.get_list_of_transformations(image)


# if __name__ == '__main__':
#     import PIL
#     img =  PIL.Image.open('/home/loxor/Documents/Humpback_Whale_Identification/data/some_imgs_from_train/0b92d79c0.jpg')
#     # listImgs, _ = TransformedImages()(img)
#     # # check all dimensions are the same
#     # c = 0
#     # for item in listImgs:
#     #     print('\t{}: {}'.format(c,item.size()))
#     #     c = c + 1
#     ret = torchvision.transforms.TenCrop((200,200))(img)
#     for x in ret:
#         print(type(x))