"""Make collage with random position, rotation, and scaling
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import glob
import argparse
import logging

import numpy as np
import skimage.transform
import skimage.io


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s',
)
LOGGER = logging.getLogger(__name__)

# Optional: try to import matplotlib if possible
HAS_PLT = False
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    LOGGER.warning('matplotlib not found. Progress visualization will be ignored...')


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        '--in_dir', type=str, default='input',
        help='Input directory',
    )
    parser.add_argument(
        '--bg_img', type=str, default='bg.jpg',
        help='Background image',
    )
    parser.add_argument(
        '--scale_min', type=float, default=0.2,
        help='Minimum scale (wrt canvas size) to rescale the images',
    )
    parser.add_argument(
        '--scale_max', type=float, default=0.6,
        help='Maximum scale (wrt canvas size) to rescale the images',
    )
    parser.add_argument(
        '--n_images', type=int, default=None,
        help='Number of images. If nothing is given, will use all images available',
    )
    parser.add_argument(
        '--canvas_h', type=int, default=1920,
        help='Height of canvas',
    )
    parser.add_argument(
        '--canvas_w', type=int, default=1080,
        help='Width of canvas',
    )
    parser.add_argument(
        '--random_flip', action='store_true',
        help='Whether to random flip the images',
    )
    parser.add_argument(
        '--out_dir', type=str, default='output',
        help='Output directory',
    )
    args = parser.parse_args()

    assert os.path.isdir(args.in_dir), '{} not found'.format(args.in_dir)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    return args


def rand_opts(canvas, scale_min, scale_max, random_flip):
    """Randomize parameters based on canvas shape and scaling range

    Args:
        cavas: allocated canvas
        scale_min: minimum scaling factor to randomize. The scale is uniformly
            picked in the range (scale_min, scale_max)
        scale_max: maximum scaling factor to randomize. The scale is uniformly
            picked in the range (scale_min, scale_max)
        random_flip: whether to randomly flip the images

    Return:
        opts: randomize parameters as a dictionary with the following keys:
            `angle`, `scale`, `offset_h`, and `offset_w`
    """
    canvas_h, canvas_w, _ = canvas.shape

    angle = np.random.uniform(low=-90, high=90)
    scale = np.random.uniform(low=scale_min, high=scale_max)
    offset_h = int(np.random.uniform(low=-canvas_h//4, high=canvas_h))
    offset_w = int(np.random.randint(low=-canvas_w//4, high=canvas_w))
    if random_flip:
        to_flip = np.random.choice([True, False])
    else:
        to_flip = False
    LOGGER.debug(
        '%f, %f-->%f, %d, %d, %r',
        angle, scale_max, scale, offset_h, offset_w, to_flip)

    opts = {
        'angle': angle,
        'scale': scale,
        'offset_h': offset_h,
        'offset_w': offset_w,
        'to_flip': to_flip,
    }
    return opts


def load_and_process(fname, canvas, scale, angle, to_flip):
    """Load and process image

    Args:
        fname: path to the image to load
        canvas: allocated canvas
        scale: scaling factor to resize the image
        angle: angle to rotate the image
        to_flip: whether to actually flip this image

    Return:
        img: processed image
    """
    canvas_h, _, _ = canvas.shape

    # Load image
    img = skimage.io.imread(fname)

    # Random flip left/right
    if to_flip:
        img = np.fliplr(img)

    # Add boundary by padding
    pad_size = max(min(img.shape[0], img.shape[1]) * 5 // 100, 1)
    img = np.pad(img, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
                 'constant', constant_values=255)

    # Resize the image by the random scale
    old_h, old_w, _ = img.shape
    new_h = int(canvas_h * scale)
    new_w = int(old_w * new_h / old_h)
    img = skimage.transform.resize(
        img, (new_h, new_w, 3),
        anti_aliasing=True, preserve_range=True)

    # Rotate the image by random angle
    img = skimage.transform.rotate(img, angle, resize=True, mode='constant')

    # Typecast back to uint8
    img = img.astype(np.uint8)
    return img


def put_in_canvas(fname, canvas, scale_min, scale_max, random_flip, viz=True):
    """Load image and randomly put in canvas

    Args:
        fname: path to the image file
        canvas: allocated canvas
        scale_min: minimum scale to resize the image
        scale_max: maximum scale to resize the image
        random_flip: whether to randomly flip the images
        viz: whether to visualize the current progress
    """
    # Prepare random parameters
    opts = rand_opts(canvas, scale_min, scale_max, random_flip)
    img = load_and_process(fname, canvas,
                           opts['scale'], opts['angle'], opts['to_flip'])

    # Put the image in canvas
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pix = img[i, j, :]

            # Avoid using black pixel (appeared after rotation)
            if pix.sum() == 0:
                continue

            # Crop the pixels outside canvas
            if (opts['offset_h']+i < 0) or \
                    (opts['offset_h']+i >= canvas.shape[0]) or \
                    (opts['offset_w']+j < 0) or \
                    (opts['offset_w']+j >= canvas.shape[1]):
                continue

            # Copy the pixel over
            canvas[opts['offset_h']+i, opts['offset_w']+j, :] = pix

    # Visualize progress
    if HAS_PLT and viz:
        plt.imshow(canvas)
        plt.draw()
        plt.pause(0.01)
    return canvas


def main():
    """Main function"""
    # Parse input arguments
    args = parse_args()

    # Retrieve the list of images
    fname_lst = glob.glob(os.path.join(args.in_dir, '*'))
    if args.n_images is None:
        args.n_images = len(fname_lst)
    while len(fname_lst) < args.n_images:
        fname_lst += fname_lst
    fname_lst = np.random.permutation(fname_lst)[:args.n_images+1]

    # Create canvas
    if os.path.isfile(args.bg_img):
        # Get the background image if given
        canvas = skimage.io.imread(args.bg_img)
        canvas = skimage.transform.resize(
            canvas, (args.canvas_h, args.canvas_w, 3),
            anti_aliasing=True, preserve_range=True)
        canvas = canvas.astype(np.uint8)
    else:
        # Otherwise use black canvas
        canvas = np.zeros([args.canvas_h, args.canvas_w, 3], dtype=np.uint8)

    # Reducing the scale_max by delta each time an image is read
    delta = (args.scale_max - args.scale_min) / len(fname_lst)
    current_scale_max = args.scale_max

    # Put all images in the canvas
    for i, fname in enumerate(fname_lst):
        LOGGER.info('Processing image: %d/%d', i+1, len(fname_lst))
        put_in_canvas(
            fname, canvas, args.scale_min, current_scale_max, args.random_flip)
        current_scale_max -= delta

        skimage.io.imsave(
            os.path.join(args.out_dir, '{}.jpg'.format(i+1)),
            canvas)

    return 0


if __name__ == '__main__':
    sys.exit(main())
