# Data generator, multi-threaded
# 多线程数据生成器，极大加快数据生成速度，并优化了内存占用。
# NOTE： 需要注意随机数生成的线程安全问题，目前没有遇到问题。
""" 
NOTE: Some of the random functions (e.g. random.gauss()) are not thread-safe, 
and would generate same values across different thread.

The functions used in this project (randint, randn) should be thread-safe.
"""

from typing import Tuple

from PIL import ImageFont, Image
from torch.functional import Tensor
from mona.config import config
import os
import pathlib
from multiprocessing import Pool
import glob

import torchvision.transforms as transforms
import torch
import datetime
from itertools import chain

from mona.datagen.datagen import DataGen
from mona.text import get_lexicon

lexicon = get_lexicon(config["model_type"])
if config["model_type"] == "Genshin":
    fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 90)]
elif config["model_type"] == "StarRail":
    fonts = [ImageFont.truetype("./assets/starrail.ttf", i) for i in range(15, 90)]
elif config["model_type"] == "WutheringWaves":
    fonts = [ImageFont.truetype("./assets/wuthering_waves/ARFangXinShuH7GBK-HV.ttf", i) for i in range(15, 90)]
datagen = DataGen(config, fonts, lexicon)


def load_extra_training_data(extra_folder="extra_training_data"):
    """Load extra training images and their corresponding labels from a folder.
    
    Args:
        extra_folder (str): Path to the extra training data folder containing images and txt files
        
    Returns:
        list: List of tuples (tensor, label) for extra training data
    """
    extra_data = []
    extra_path = pathlib.Path(extra_folder)
    
    if not extra_path.exists():
        return extra_data
    
    # Find all image files in the extra training data folder
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(extra_path / ext)))
    
    for image_file in image_files:
        # Get corresponding txt file
        txt_file = pathlib.Path(image_file).with_suffix('.txt')
        
        if txt_file.exists():
            try:
                # Load image and convert to tensor
                im = Image.open(image_file).convert('L')  # Convert to grayscale
                
                # Check image dimensions
                if im.size != (384, 32):
                    print(f"Warning: {image_file} has incorrect size {im.size}, expected (384, 32). Skipping.")
                    continue
                
                tensor = transforms.ToTensor()(im)
                tensor = torch.unsqueeze(tensor, dim=0)
                
                # Load label from txt file
                with open(txt_file, 'r', encoding='utf-8') as f:
                    label = f.read().strip()
                
                extra_data.append((tensor, label))
                
            except Exception as e:
                print(f"Warning: Failed to load {image_file}: {e}")
    
    print(f"Loaded {len(extra_data)} extra training samples from {extra_folder}")
    return extra_data


def progressBar(current, total, barLength=40):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def fill_data(target_tensor_slice: Tensor) -> list:
    """Fill a given tensor slice with generated image

    Args:
        target_tensor_slice (Tensor): Tensor slice to store the generated image

    Returns:
        list: List of labels
    """
    length = target_tensor_slice.shape[0]
    y = []
    for i in range(length):
        im, text = datagen.generate_image()
        tensor = transforms.ToTensor()(im)
        tensor = torch.unsqueeze(tensor, dim=0)
        # NOTE: here tensor.shape == [1, 1, 32, 384]
        target_tensor_slice[i] = tensor
        y.append(text)

        # Worker print progress
        if i % 100 == 0:
            progressBar(i, length)
    return y


def gen_dataset_with_label(size, threads=2, extra_folder="extra_training_data") -> Tuple[Tensor, list]:
    # Load extra training data first
    extra_data = load_extra_training_data(extra_folder)
    extra_count = len(extra_data)
    
    # Calculate how many generated samples we need
    generated_size = max(0, size - extra_count)
    
    if extra_count > size:
        print(f"Warning: Extra training data ({extra_count}) exceeds requested size ({size}). Using first {size} samples.")
        extra_data = extra_data[:size]
        extra_count = size
        generated_size = 0
    
    # Allocate the output Tensor
    x = torch.zeros((size, 1, 32, 384))
    all_labels = []
    
    # Generate regular data first
    if generated_size > 0:
        # Split tensor for parallel generation
        generated_x = x[:generated_size]
        x_split = torch.tensor_split(generated_x, threads, dim=0)
        
        with Pool(threads) as p:
            print(f"Starting threadpool with {threads} threads for {generated_size} generated samples.")
            labels = p.map(fill_data, x_split)
            print("\nStopping threadpool.")
            all_labels.extend(list(chain.from_iterable(labels)))
    
    # Fill with extra training data at the end (higher priority)
    for i, (tensor, label) in enumerate(extra_data[:size]):
        x[generated_size + i] = tensor
        all_labels.append(label)
    
    return x, all_labels


if __name__ == '__main__':
    train_size = config["train_size"]
    validate_size = config["validate_size"]

    folder = pathlib.Path("data")
    if not folder.is_dir():
        os.mkdir(folder)

    # Use physical cores only
    threads = max(1, os.cpu_count())
    threads = 4

    print(
        f"Train size {train_size}, Val size {validate_size}, Thread count {threads}")

    # Generate and save training set
    print(f"{datetime.datetime.now()} Generating training data")
    x, y = gen_dataset_with_label(size=train_size, threads=threads)

    print(f"{datetime.datetime.now()} Saving training data")
    torch.save(x, "data/train_x.pt")
    torch.save(y, "data/train_label.pt")
    del x, y

    print(f"{datetime.datetime.now()} Generating validation data")
    x, y = gen_dataset_with_label(size=validate_size, threads=threads)

    print(f"{datetime.datetime.now()} Saving validation data")
    torch.save(x, "data/validate_x.pt")
    torch.save(y, "data/validate_label.pt")

    # Verify the result
    # for tensor, y in zip(x,y):
    #     arr = tensor.squeeze()
    #     im = Image.fromarray(np.uint8(arr * 255))
    #     im.show()
    #     print(y)
    #     import time
    #     time.sleep(1)
