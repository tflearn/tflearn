'''
this file is used to gen the all data info(data name and class) in your data folder
and is user for image_preloader with mode='file'
'''
import os
from PIL import Image
import numpy as np


def gen_files_list(data_path, write_file):
    samples = []
    targets = []
    label = 0
    try:  # Python 2
        classes = sorted(os.walk(data_path).next()[1])
    except Exception:  # Python 3
        classes = sorted(os.walk(data_path).__next__()[1])
    for c in classes:
        c_dir = os.path.join(data_path, c)
        try:  # Python 2
            walk = os.walk(c_dir).next()
        except Exception:  # Python 3
            walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            samples.append(os.path.join(c_dir, sample))
            targets.append(label)
        label += 1
    with open(write_file, "w") as fwrite:
        for index, i in enumerate(samples):
            try:
                img = Image.open(i)
                img.load()
                img = np.asarray(img)
                # filter the images which the channel is not 3
                if img.shape[2] == 3:
                    line = i + ' ' + str(targets[index]) + '\n'
                    fwrite.write(line)
            except:
                continue
            fwrite.write(line)


if __name__ == '__main__':
    data_dir = '/data/to/your/train/data'
    write_files_name = "files_list"
    gen_files_list(data_dir, write_files_name)
