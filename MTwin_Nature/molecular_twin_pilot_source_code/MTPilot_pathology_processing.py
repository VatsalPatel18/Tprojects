# Copyright (C) 2022 Betteromics Inc. - All Rights Reserved

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Copyright (C) 2022 - Betteromics Inc.
# # Preprocess and generate tiles from raw SVS (histopathology) images

# %load_ext autoreload
# %autoreload 2

# +
from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import glob, h5py, imageio, multiprocessing, os, pathlib
import matplotlib.pyplot as plt

from csbdeep.utils import Path, normalize

from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import Config2D, StarDist2D

from PIL import Image
# %matplotlib inline

INPUT_SLIDE_COUNT = None # None to process all available slides.
OUTPUT_INTERMEDIATE_ARTIFACTS = False  # Enable to emit individual tiles and the corresponding tile masks.
NUM_CPU_PARALLEL = multiprocessing.cpu_count()


# +
def get_input_slide_list(input_slide_path, processed_slide_path):
    input_slides = glob.glob(input_slide_path)
    dict_slide_paths = { pathlib.Path(i).stem : i for i in input_slides }

    processed_slides = glob.glob(processed_slide_path)
    dict_processed_slides = { pathlib.Path(i).stem : i for i in processed_slides }

    slides_to_process = {x:dict_slide_paths[x] for x in dict_slide_paths  
                                                if x not in dict_processed_slides}
    return list(slides_to_process.values())

def gputools_available():
    try:
        import gputools
    except:
        return False
    return True

def check_label_exists(label, label_map):
    ''' Checking if a label is a valid label. 
    '''
    if label in label_map:
        return True
    else:
        print("Provided label " + str(label) + " not present in label map.")
        print("Setting label as -1 for UNRECOGNISED LABEL.")
        print(label_map)
        return False

def generate_label(regions, region_labels, point, label_map):
    ''' Generates a label given an array of regions.
        - regions               array of vertices
        - region_labels         corresponding labels for the regions
        - point                 x, y tuple
        - label_map             the label dictionary mapping string labels to integer labels
    '''
    for i in range(len(region_labels)):
        poly = Polygon(regions[i])
        if poly.contains(Point(point[0], point[1])):
            if check_label_exists(region_labels[i], label_map):
                return label_map[region_labels[i]]
            else:
                return -1
    # By default, we set to "Normal" if it exists in the label map.
    if check_label_exists('Normal', label_map):
        return label_map['Normal']
    else:
        return -1

def get_regions(path):
    ''' Parses the xml at the given path, assuming annotation format importable by ImageScope. '''
    xml = minidom.parse(path)
    # The first region marked is always the tumour delineation
    regions_ = xml.getElementsByTagName("Region")
    regions, region_labels = [], []
    for region in regions_:
        vertices = region.getElementsByTagName("Vertex")
        attribute = region.getElementsByTagName("Attribute")
        if len(attribute) > 0:
            r_label = attribute[0].attributes['Value'].value
        else:
            r_label = region.getAttribute('Text')
        region_labels.append(r_label)

        # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
        coords = np.zeros((len(vertices), 2))

        for i, vertex in enumerate(vertices):
            coords[i][0] = vertex.attributes['X'].value
            coords[i][1] = vertex.attributes['Y'].value

        regions.append(coords)
    return regions, region_labels

def tissue_percent(np_img, background_pixel_threshold = 230):
    """
    Determine the percentage of a NumPy array that is tissue (how many of the values are non-zero/background values).
    Args:
      np_img: Image as a NumPy array.
    Returns:
      The percentage of the NumPy array that is tissue (non-background).
    """
    np_img = np.where(np_img < background_pixel_threshold, np_img, 0)
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        tissue_percentage = np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        tissue_percentage = np.count_nonzero(np_img) / np_img.size * 100
    return tissue_percentage

# OpenSlide uses interior tile size without overlap, while
# we use patch_size to be the actual tile pixel dimension.
def patch_to_tile_size(patch_size, overlap):
    return patch_size - overlap*2

def generate_stardist_seg(stardist_model, input_tile, seg_file_base):
    axis_norm = (0,1)   # normalize channels independently
    
    # Normalize and predict on input tile.
    norm_tile = normalize(input_tile, 1, 99.8, axis=axis_norm)
    labels, details = stardist_model.predict_instances(norm_tile)
    
    # Save the original image and label/mask
    if OUTPUT_INTERMEDIATE_ARTIFACTS:
        Image.fromarray(labels).save(seg_file_base + "mask.png")
        export_imagej_rois(seg_file_base + 'rois.zip', details['coord'])
    
    return labels
    
def save_tiles_to_disk(output_dir, patches, coords, file_name, labels):
    """ Saves numpy patches to .png files (full resolution). 
        Meta data is saved in the file name.
        - output_dir        folder to save images in
        - patches           numpy images
        - coords            x, y tile coordinates
        - file_name         original source WSI name
        - labels            patch labels (opt)
    """
    os.makedirs(output_dir, exist_ok=True)
    save_labels = len(labels)
    for i, patch in enumerate(patches):
        # Construct the new PNG filename
        patch_fname = file_name + "_" + str(coords[i][0]) + "_" + str(coords[i][1]) + "_"

        if save_labels:
            patch_fname += str(labels[i])

        # Save the image.
        Image.fromarray(patch).save(output_dir + patch_fname + "tile.png")

def sample_and_store_patches(file_path, pixel_overlap, patch_size=1024, level=None,
                             xml_dir=False, label_map={}, limit_bounds=True,
                             tissue_pct_threshold=None, stardist_model=None,
                             rows_per_txn=20, output_dir=''):
    ''' Sample patches of specified size from .svs file.
        - file_path             path of whole slide image to sample from
        - pixel_overlap         pixels overlap on each side
        - level                 0 is lowest resolution; level_count - 1 is highest
        - xml_dir               directory containing annotation XML files
        - label_map             dictionary mapping string labels to integers
        - tissue_pct_threshold  filter out tiles containing less than tissue pct threshold
        - stardist_model        run StarDist tile segmentation on tiles if set
        - rows_per_txn          how many patches to load into memory at once
        - output_dir            folder to save images in
        Note: patch_size is the dimension of the sampled patches, NOT equivalent to openslide's definition
        of tile_size. This implementation was chosen to allow for more intuitive usage.
    '''
    tile_size = patch_to_tile_size(patch_size, pixel_overlap)
    slide = open_slide(file_path)
    tiles = DeepZoomGenerator(slide, tile_size=tile_size,
                              overlap=pixel_overlap, limit_bounds=limit_bounds)
    base_file_name = pathlib.Path(file_path).stem
    seg_output_dir = output_dir + "seg/"
    os.makedirs(seg_output_dir, exist_ok=True)
    
    if xml_dir:
        # Expect filename of XML annotations to match SVS file name
        regions, region_labels = get_regions(base_file_name + ".xml")

    if level == None:
        # If not set, default to highest resolution level for each slide.
        level = tiles.level_count - 1
    
    if level >= tiles.level_count:
        print("Requested level does not exist. Number of slide levels: " + str(tiles.level_count))
        return 0

    x_tiles, y_tiles = tiles.level_tiles[level]
    x_pixels, y_pixels = tiles.level_dimensions[level]

    x, y = 0, 0
    count, batch_count, del_count = 0, 0, 0
    patches, coords, labels = [], [], []
    
    print("Processing " + base_file_name + ", at level:" + str(level) + " with x_tiles, y_tiles: " +
          str(x_tiles) + ", " + str(y_tiles) + " at tile_size: " + str(patch_size))
    
    # Initialize empty segmentation mask for slide
    slide_mask_stitched = np.zeros((y_pixels, x_pixels), dtype=np.uint8)
    
    while y < y_tiles:
        while x < x_tiles:
            new_tile = np.array(tiles.get_tile(level, (x, y)), dtype=np.uint8)
            # OpenSlide calculates overlap in such a way that sometimes depending on the dimensions, edge
            # patches are smaller than the others. We will ignore such patches.
            if np.shape(new_tile) == (patch_size, patch_size, 3):
                # Optionally filter tiles for minimum tissue percentage
                if (tissue_pct_threshold != None and tissue_percent(new_tile) > tissue_pct_threshold):
                    patches.append(new_tile)
                    coords.append(np.array([x, y]))
                    count += 1
                    # Optionally generate StarDist segmentation masks inline
                    if stardist_model:
                        seg_file_base = seg_output_dir + base_file_name + "_" + str(x) + "_" + str(y) + "_"
                        tile_mask_np = generate_stardist_seg(stardist_model, new_tile, seg_file_base)
                        # Stitch together output segmentations into a slide mask in HDF5
                        slide_mask_stitched[y*patch_size:(y+1)*patch_size,
                                            x*patch_size:(x+1)*patch_size] = tile_mask_np
                else:
                    del_count += 1

                # Calculate the patch label based on centre point.
                if xml_dir:
                    converted_coords = tiles.get_tile_coordinates(level, (x, y))[0]
                    labels.append(generate_label(regions, region_labels, converted_coords, label_map))
            x += 1

        # To save memory, we will save data into the dbs every rows_per_txn rows. i.e., each transaction will commit
        # rows_per_txn rows of patches. Write after last row regardless.
        if (y % rows_per_txn == 0 and y != 0) or y == y_tiles-1:
            if OUTPUT_INTERMEDIATE_ARTIFACTS:
                save_tiles_to_disk(output_dir + "tiles/", patches, coords, base_file_name, labels)
            del patches
            del coords
            del labels
            patches, coords, labels = [], [], [] # Reset right away.

        y += 1
        x = 0
    
    # If StarDist model segmentation was applied on the slide, output resulting stitched mask as HDF5 file.
    if stardist_model:
        mask_output_dir = output_dir + "masks/"
        os.makedirs(mask_output_dir, exist_ok=True)
        # Store stitched segmentation mask as an HDF5 file.
        hf_file_name = mask_output_dir + base_file_name + ".hdf"
        hf = h5py.File(hf_file_name, 'a') # open a hdf5 file
        dset = hf.create_dataset('default', data=slide_mask_stitched)  # write the data to hdf5 file
        hf.close()  # close the hdf5 file
        print('Stitiched mask for ' + base_file_name + ', hdf5 file size: %d bytes'%os.path.getsize(hf_file_name))
        
        # Store stitched segmentation mask as a tiff file.
        #if OUTPUT_INTERMEDIATE_ARTIFACTS:
        #    tiff_file_name = seg_output_dir + base_file_name + ".tif"
        #    im = Image.fromarray(slide_mask_stitched)
        #   im.save(tiff_file_name)
    print("Finished processing " + base_file_name + ", num_tiles_processed:" + str(count) + ", num_tiles_dropped: " + str(del_count))
    return (count, del_count)


# +
# For each input SVS slide in the input directory
# 1) Load SVS input slide
# 2) Segment slide into X tiles of NxN dimensions at Z zoom level (or max available zoom if unset)
# 3) Drop tiles containing less than specified tissue threshold
# 4) Directly call StarDist model inline for each generated tile 
# 5) Save generated artifact files in output directory, in png format
# 6) Stitch together StarDist segments into unified HDF5 mask

# Set parameters for input file directory of SVS images, patch size, level (magnification), and overlap.
patch_size = 1024 # NxN pixel dimension of segmented tiles
level = None # None or 0-17 typically for slide magnification level.
overlap = 0 # pixel overlap between segmented tiles.
tissue_pct_thres = 30 # min tissue percent on segmented tile.
input_slide_dir = "/home/ubuntu/wsi/svs/"
level_str = "max"
if level:
    level_str = str(level)
output_dir = "/home/ubuntu/wsi/processed_tiles/output_" + level_str + "_" + str(patch_size) + "/"
os.makedirs(output_dir, exist_ok=True)
xml_dir = input_slide_dir

# All possible labels mapped to integer ids in order of increasing severity.
label_map = {'Normal': 0,
             'QC': 0,
             'pap': 1,
             'nonpap': 2
            }

# Load and configure StarDist model.
config = Config2D(
    axes         = 'YXC', # expect YxX image input with C channels
    n_rays       = 32, # Rays to use when tracing out segmentation shape.
    grid         = (2,2), # Predict on subsampled grid for increased efficiency and larger field of view
    use_gpu      = False and gputools_available(),
    n_channel_in = 3,
)
stardist_model = StarDist2D.from_pretrained('2D_versatile_he')
stardist_model.config = config

def stardist_seg_task(input_slide_path, pbar):
    count, del_count = sample_and_store_patches(input_slide_path, pixel_overlap=overlap, patch_size=patch_size, level=level,
                                                xml_dir=False, label_map={}, limit_bounds=True,
                                                tissue_pct_threshold=tissue_pct_thres, stardist_model=stardist_model,
                                                rows_per_txn=2, output_dir=output_dir)
    pbar.update(1)
    return count

def stardist_seg_worker(input_slide_paths):
    for infile in input_slide_paths:
        count, del_count = sample_and_store_patches(infile, pixel_overlap=overlap, patch_size=patch_size, level=level,
                                                    xml_dir=False, label_map={}, limit_bounds=True,
                                                    tissue_pct_threshold=tissue_pct_thres, stardist_model=stardist_model,
                                                    rows_per_txn=2, output_dir=output_dir)


# -

slide_paths_to_process = get_input_slide_list(input_slide_dir + "*.svs", output_dir + "masks/*.hdf" )[:INPUT_SLIDE_COUNT]
sharded_slide_paths = np.array_split(slide_paths_to_process, NUM_CPU_PARALLEL)
print("Processing slides: " + str(sharded_slide_paths))

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
if __name__ == '__main__':
    with tqdm(total=len(slide_paths_to_process)) as pbar:
          with ThreadPoolExecutor(max_workers=NUM_CPU_PARALLEL) as ex:
                futures = [ex.submit(stardist_seg_task, slide_path, pbar) for slide_path in slide_paths_to_process]
                for future in as_completed(futures):
                       result = future.result() 


# +
np.random.seed(6)
lbl_cmap = random_label_cmap()
axis_norm = (0,1)   # normalize channels independently

def render_tile_example(model, tile, show_dist=True):
    img = normalize(tile, 1, 99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    plt.figure(figsize=(13,10))
    img_show = img if img.ndim==2 else img[...,0]
    coord, points, prob = details['coord'], details['points'], details['prob']
    #plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    plt.subplot(121); plt.imshow(tile); plt.axis('off')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    a = plt.axis()
    _draw_polygons(coord, points, prob, show_dist=show_dist)
    plt.axis(a)
    #plt.subplot(122); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    plt.subplot(122); plt.imshow(tile); plt.axis('off')
    plt.tight_layout()
    plt.show()


# +
# Generate visualization of input slides, output tiles, filtered tiles and output segmentations
tile_fnames = glob.glob(output_dir + "tiles/*.png")
tiles = list(map(np.array, map(imageio.imread, tile_fnames[0:10])))

for tile in tiles:
    render_tile_example(stardist_model, tile)

# +
# Show saved slide tiles and corresponding masks
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import glob, pathlib

display(HTML("<style>.container { width:100% !important; }</style>"))

NUM_ROWS = 9
tile_fnames = np.array(sorted(glob.glob(base_dir + "tiles/*.png")))
seg_fnames = np.array(sorted(glob.glob(base_dir + "seg/*.png")))

shuffler = np.random.permutation(len(tile_fnames))
tiles_shuffled = tile_fnames[shuffler]
seg_shuffled = seg_fnames[shuffler]

f, ax_arr = plt.subplots(NUM_ROWS, 3, figsize=(24,72))

for i in range(NUM_ROWS):
    ax_arr[i, 0].imshow(plt.imread(tiles_shuffled[i]))
    ax_arr[i, 0].set_title(f'{pathlib.Path(tiles_shuffled[i]).stem}')
    ax_arr[i, 1].imshow(plt.imread(seg_shuffled[i]))
    ax_arr[i, 1].set_title(f'{pathlib.Path(seg_shuffled[i]).stem}')
    ax_arr[i, 2].imshow(plt.imread(tiles_shuffled[i]))
    ax_arr[i, 2].imshow(plt.imread(seg_shuffled[i]), alpha=0.5)
plt.show()

# +
# Inspect all HDF5 generated image masks for slides
# %matplotlib ipympl

import h5py
from hdfviewer.widgets.HDFViewer import HDFViewer

hdf_fnames = np.array(sorted(glob.glob(base_dir + "seg/*.hdf")))

for hdf_file in hdf_fnames:
    hdf5 = h5py.File(hdf_file, "r")
    display(HDFViewer(hdf5))

