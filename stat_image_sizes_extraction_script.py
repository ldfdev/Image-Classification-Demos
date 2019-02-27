import os
from PIL import Image

big_num = 2**100
min_height, min_width, max_height, max_width = [big_num,big_num,0,0]
avg_width, avg_height, num_of_imgs = [0,0,0]
DATA_PATH = '/home/loxor/Documents/Humpback_Whale_Identification/data/train/'
save_file = open('image_stats.csv','a')
save_file.write('IMG_name,IMG_width,IMG_height\n')
for f in os.listdir(DATA_PATH):
    if os.path.isfile(os.path.join(DATA_PATH, f)):
        num_of_imgs += 1
        img = Image.open(os.path.join(DATA_PATH,f))
        width, height = img.size
        save_file.write('{0},{1},{2}\n'.format(f,width,height))
        # if width < min_width:
        #     min_width = width
        # elif width > max_width:
        #     max_width = width
        # if height > max_height:
        #     max_height = height
        # elif height < min_height:
        #     min_height = height
        # avg_width += width
        # avg_height += height
save_file.close()
print('Minimum width {width}\nMinimum height{height}\nAverage Dimensions(weight, height) {avg_width}x{avg_height}'.format\
    (width=min_width, height = min_height, avg_width=avg_width//num_of_imgs, avg_height=avg_height//num_of_imgs))