import tensorflow as tf

#For time stamps
import time
from datetime import datetime
from zoneinfo import ZoneInfo
        

#Report on if script is training with GPU or not
print("Getting GPU Info...")
print("GPU - " + str(tf.config.list_physical_devices('GPU')));
gpu_count = len(tf.config.list_physical_devices('GPU'))
print("Number of GPUs: " + str(gpu_count))
if(gpu_count == 0):
    print("WARNING! NO GPUs WILL BE USED!")
else:
    print("Using GPU for training...")

#Allow for memory growth - prevent out of memory error
for gpu in tf.config.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)

"""
#https://forums.developer.nvidia.com/t/out-of-memory-error-from-tensorflow-any-workaround-for-this-or-do-i-just-need-a-bigger-boat/75082
#Set memory usage / allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
"""

#DEPENDENCY LOADER

#Import numpy for math functions
import numpy as np
import random


#Switched to Keras
import keras
from keras import layers, regularizers

#Import bounding box and visualization modules for keras for image and bb handling
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

from keras.models import Model

from keras.metrics import MeanIoU

import csv as csv

from enum import Enum                           #Enumerations

import os                                     #For image processing and listing out files in the Collab mounted drive

from PIL import Image                         #Pillow (PIL) allows creating new png images e.g. seg masks from the predicted data...

#-----------------------------------#
#Enum
#-----------------------------------#

#The type of model used.
#   YOLO involves object detection and drawing bounding boxes around predicted objects
#   UNET involves semantic segmentation in this case - making pixel-by-pixel predictions based on training masks provided to the model for training
#   YOLO_SEG - Means calculating YOLO bounding boxes from the segmentation images output from the UNET mask predictions. RUN UNET VALIDATION FIRST TO GENERATE THOSE IMAGES!
#       Main distinction between YOLO and YOLO_SEG - both train on the mask data. However, YOLO_SEG validates on the output mask data from the unet model, where as YOLO just validates on existing masks...
#       Results for ModelType.YOLO will be better because it's not a calculated mask - it's the mask from roboflow, but, they should not be used to determine anything other than the YOLO model's 
#               ability to understand existing data masks (not predicted data)!
class ModelType(Enum):
  YOLO = 0
  UNET = 1
  YOLO_SEG = 2

#A type of path, e.g. for models, images, etc.
class PathType(Enum):
  MODEL = 0
  IMAGES = 1
  CHECKPOINT = 2

#The desired type of image input, e.g. RGB images or masks
class ImageDataType(Enum):
  RGB = 0
  MASKS = 1

#The size of the yolo model used
#Taken from tut https://keras.io/examples/vision/yolov8/ (the yolo sizes, not enum values)
class ModelSize(Enum):
  XtraSmall = 0
  Small = 1
  Medium = 2
  Large = 3
  XtraLarge = 4

#Different optimizer options
class Optimizer(Enum):
  ADAM = 0
  RMSPROP = 1

#Image color channels to use for image classification
class ImageColorChannelMode(Enum):
  RGB = 0
  GREYSCALE = 1
  RGBA = 2

#Should this code run training or the demo?
class Mode(Enum):
  TRAIN = 0
  DEMO = 1

#IOU Metric Types
class MetricType(Enum):
  AVGIOU = 0
  CLASSACCURACY = 1
  BOXACCURACY = 2
  PIXELACCURACY = 3
  CLASSPRECISION = 4
  BOXPRECISION = 5
  PIXELPRECISION = 6
  CLASSRECALL= 7
  BOXRECALL= 8
  PIXELRECALL = 9
  CLASSF1 = 10
  BOXDICE = 11
  PIXELDICE = 12

#Represents all the possible cycle states this model can be in
class CycleType(Enum):
  TRAINING = 0
  VALIDATION = 1
  TESTING = 2

#What should the YOLO model train on? Prefabed masks from roboflow, or UNet predictions. Both use the same bounding boxes
class YOLOTrainingSource(Enum):
  PrefabMasks = 0
  UNetPredictions = 1
  OriginalImages = 2


#-----------------------------------#
#Properties
#-----------------------------------#


#Application mode
#Mode.TRAIN - For training a new model. Saves to the model name set
#Mode.DEMO - Once the model is trained, for getting image selections and predictions
#MODE = Mode.TRAIN
MODE = Mode.DEMO

#Select the specific model type - YOLO (bounding box segmentation) or U-NET Xception-Like Model
#MODEL_TYPE = ModelType.YOLO
#MODEL_TYPE = ModelType.UNET
MODEL_TYPE = ModelType.YOLO_SEG

#The size of the YOLO model used...
#MODEL_SIZE = ModelSize.Small
MODEL_SIZE = ModelSize.Medium
#MODEL_SIZE = ModelSize.Large

#If true, saves segmentation image predictions to file. If false, skips this step, assuming this had already been done previously, as a shortcut.
#Saving images takes time, but, the output of seg images isn't necessarily deterministic (that I know of), so, this would be True for anything but code debugging (where we want to save time)
OPTION_SAVE_SEG_IMAGES = False

#If true, will calculate metrics and save them to file. If false, will skip this step all together
OPTION_CALCULATE_METRICS = True

#If true, will show the correct output images. If false, will not. Skipping showing the output images will just cause the skip to run and end without a break from the image viewer.
OPTION_SHOW_OUTPUT_IMAGES = True

#Output log on YOLO matched bounding boxes entries to their images
REPORT_YOLO_FILE_MATCHES = False

#Before training, show training image gallery?
OPTION_SHOW_TRAINING_IMAGE_GALLERY = False

#If true, the prediction model for unet will generate output mask predictions for every single image in the all data. Will run testing and validation data and save images for those.
OPTION_GENERATE_FULL_YOLOSEG_OUTPUTMASK_SELECTION = False

#What should the source of YOLO training images be? Prefab masks or UNet predictions? Recommend using predictions
YOLO_TRAINING_SOURCE = YOLOTrainingSource.OriginalImages

#TODO: MAYBE DO THIS ON FINAL EPOCH? - WILL SAVE ALL VALIDATION IMAGES AS WELL TO FILE
#When training with image segmentation, save progressive image validation predictions to file
#TRAINING_SAVE_PROGRESSIVE_IMAGES = True

#This sets the size based on the YOLO version used
#I would like to make this functional in the future
#Unfortunately, only YOLOv8 versions work at the moment
#The file width and height to convert images to in processing
FILESIZE = 512

#The learning rate for all models
LEARNING_RATE=0.001

#Prevents exploding gradients
GLOBAL_CLIPNORM = 10.0

#The weight decay to use in the optimizer.
#Set this to low to have a less chaotic impact during training
#0.05 Seems like an optimal result
WEIGHT_DECAY = 0.05

#Number of input channels to use (RGB = 3 channels)
INPUT_CHANNELS = 3

#Colors for bounding mask display - will alternate between these
BOUNDING_BOX_DISPLAY_COLORS = ((1.0,0,0), (0,1.0,0), (0,0,1.0), (1.0,1.0,0), (0,1.0,1.0), (1.0,0,1.0), (0.5,.125,.25), (.25,.125,.5), (.5,.25,.5))

#The optimizer selection for this YOLO instance
#Can choose ADAM or RMSProp
OPTIMIZER = Optimizer.ADAM

#Numb
#Number of training epochs (loops) to run
EPOCHS = 50

#Should the trainer show extra status text?
VERBOSE = 1

#The number of batches to run in one epoch
BATCH_SIZE = 4

#Maximum bounding boxes per image
MAX_ANCHOR_BOXES = 64

#Scoring IOU Threshold - typically 50% per https://zihaogeng.medium.com/how-to-evaluate-an-object-detection-model-iou-precision-recall-and-map-f7cc12e0dcf6
#All scores below this will be considered "wrong." All scores greater than or equal to this will be considered "right"
IOU_SCORING_THRESHOLD = 0.5

#The percent of data to use as test data instead of training data
#Recommendations: Test 20%, Validation 5% of test data. This are MY recommendations based on class, not a tutorial or something like that's
TEST_DATA_PCT = 0.1
VALIDATION_SPLIT_PCT = 0.05

#TODO: USE DIFFERENT SEGMENTS OF WATER...
#Class List - Updated to segment water into different sizes for better results
#I would like to expand this model to a three class problem instead of one

if(MODEL_TYPE == ModelType.YOLO or MODEL_TYPE == ModelType.YOLO_SEG):
  CLASS_LIST = ("Lake","River")
else:
  CLASS_LIST = ("Nothing","Lake","River")
class_mapping = dict(zip(range(len(CLASS_LIST)), CLASS_LIST))

#Number of classes in this problem. Though we're looking at water vs. not water, we're really just looking for water, so, one class.
# Non-water is just "null" in a logical sense and not a class.
class_count = len(CLASS_LIST)

#---IMAGE PROCESSING STUFF... ---#
#Fetch RGB channels and setr alpha as 1 (as, none of the imported images have an alpha (transparency) channel anyways)
IMAGE_COLOR_CHANNEL_MODE = ImageColorChannelMode.RGB

#The size of edges to cut off each processed image
#This is data augmentation to accomodate for issues with the training data. It had weird black stripes on the edge of it.
IMAGE_CUT_EDGES_SIZE = 4

#Data Augmentation Properties
AUGMENTATION_FLIP_VERTICALLY = True
AUGMENTATION_FLIP_HORIZONTALLY = True
AUGMENTATION_RANDOM_GAUSSIAN_BLUR = False
AUGMENTATION_RANDOM_ROTATION = False
AUGMENTATION_RANDOM_COLORJITTER = False

#Represents a bounding box on an image
#x - The x coordinate of the center of this bounding box
#y - The y coordinate of the center of this bounding box
#w - The width of this bounding box
#h - The height of this bounding box
class BoundingBox():

  imgIndex = -1
  x0 = 0
  y0 = 0
  x1 = 0
  y1 = 0
  classification = 0

  def __init__(self, imgIndex, classification, x0,y0, x1,y1):
      self.imgIndex = imgIndex
      self.classification = classification
      self.x0 = x0
      self.y0 = y0
      self.x1 = x1
      self.y1 = y1

  #Output in tensor format
  def toTensor(self):

      #1.0 = Full probability water is here, x center coor, y center coor, width, height, 1 flag - water is here
      return [self.x0, self.y0, self.x1, self.y1]

#Reads the given text file with the given filename
def read_text_file(file_name):
  try:
    print("Reading text file " + file_name)
    with open(file_name) as file:
      lines = file.readlines()
      file.close()
      return lines      
  except Exception as e:
    print("ERROR! Could not read the following file during training: " + file_name)
    return None

#Loads an image during training, validation, or testing for either model type
#image_path - The path of the image being loaded
#channels - Number of channels to load for this image
def load_image(image_path, channels):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=channels)
    return image

#Given the image path and mask path for a row, loads the image with its mask image. Runs during any model processing.
def load_seg_data(image_path, mask_path):

  #Load the base rgb image being trained on (3 channels, rgb)
  image = load_image(image_path, 3)

  #Load the mask for the base image (1 channel, grey)
  mask = load_image(mask_path, 1)

  #Resizing for all images
  img_resizer = keras.layers.Resizing(height=FILESIZE, width=FILESIZE, interpolation="nearest")
  mask_resizer = keras.layers.Resizing(height=FILESIZE, width=FILESIZE, interpolation="nearest")
  image = img_resizer(image)
  mask = mask_resizer(mask)

  return {"images": tf.cast(image, tf.float32), "segmentation_masks": tf.cast(mask, tf.float32)}

#Given the image path, classes, and bounding boxes for a row, loads the image and then combines the image, and bounding boxes. Runs during any model processing.
#Class is packed into the bounding box record
def load_yolo_data(image_path, classes, bbox):
  
  # Read Image
  image = load_image(image_path, 3)
  bounding_boxes = {
      "classes": tf.cast(classes, dtype=tf.float32),
      "boxes": bbox,
  }
  return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

#Converts from dictionary type to tuple type
#https://keras.io/examples/vision/yolov8/
#inputs - The incoming dictionary
def yolo_dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=MAX_ANCHOR_BOXES
    )

#Converts the incoming dictionary into a tuple object
#inputs - The incoming dictionary
def seg_dict_to_tuple(inputs):
  return inputs["images"], inputs["segmentation_masks"]
    

#Write text to a file
#epoch - The current epoch if applicable. Otherwise set as -1
#logstr - The log string to write to file
#filename - The full file patht o write to
def write_file(epoch, logstr, file_path):
  try:
      with open(file_path, 'a') as file:
        file.write(str(epoch) + "," + logstr + '\n')
  except Exception as e:
      print("Could not write epoch to file : " + str(epoch))

#Gets the string descriptor for a metric type
#metric_type - The metric type to get the descriptor for
#returns - The label for a metric type (e.g. box_accuracy_)
def get_metric_type_descriptor(metric_type):
  descriptor = "total_iou_"
  if(metric_type == MetricType.AVGIOU):
    descriptor = "avg_iou_"
  elif (metric_type == MetricType.CLASSACCURACY):
    descriptor = "class_accuracy_"
  elif (metric_type == MetricType.BOXACCURACY):
    descriptor = "box_accuracy_"
  elif (metric_type == MetricType.PIXELACCURACY):
    descriptor = "pixel_accuracy_"
  elif (metric_type == MetricType.CLASSPRECISION):
    descriptor = "class_precision_"
  elif (metric_type == MetricType.BOXPRECISION):
    descriptor = "box_precision_"
  elif (metric_type == MetricType.PIXELPRECISION):
    descriptor = "pixel_precision_"
  elif (metric_type == MetricType.CLASSRECALL):
    descriptor = "class_recall"
  elif (metric_type == MetricType.BOXRECALL):
    descriptor = "box_recall"
  elif (metric_type == MetricType.PIXELRECALL):
    descriptor = "pixel_recall"
  elif (metric_type == MetricType.CLASSF1):
    descriptor = "class_f1_"
  elif (metric_type == MetricType.BOXDICE):
    descriptor = "box_dice_"
  elif (metric_type == MetricType.PIXELDICE):
    descriptor = "pixel_dice_"
  return descriptor


#Gets the string descriptor for a cycle type (e.g. training, validation, testing)
#cycle_type - The cycle type to get the descriptor for, type of CycleType
#returns - The label for a metric type (e.g. training_)
def get_cycle_type_descriptor(cycle_type):

  if(cycle_type == CycleType.TRAINING):
    return "training_"
  elif(cycle_type == CycleType.VALIDATION):
    return "validation_"
  else:
    return "testing_"


#Gets the string descriptor for a model type (e.g. training_ or testing_)
#model_type - The model type, type of ModelType
#returns - The label for a model type (e.g. training_)
def get_model_type_descriptor(model_type):
  if(model_type == ModelType.YOLO):
    return "yolo_"
  elif(model_type == ModelType.UNET):
    return "unet_"
  else:
    return "yolo_seg"

  
    

    

#Get the metric file path from the metric type and cycle type
#model_type - The type of model being run
#metric_type - Type of metric used (type of MetricType)
#cycle_type - What cycle is this in? Training? Validation? Testing?
#iou (Optional) - If this metric file is for a specific IoU threshold. If set to -1, is ignored
def get_metric_file_path(model_type, metric_type, cycle_type):

  #Base path where all metric logs get saved to...
  #metric_base_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/logs/"
  metric_base_path = "C:\\Phee\\Class\\Capstone\\metrics\\"

  #Get the descriptive value of the model type (yolo, unet, or yoloseg)
  model = get_model_type_descriptor(model_type)

  #Get the prefix for the current cycle being run (training, validation, or testing)
  prefix = get_cycle_type_descriptor(cycle_type)

  #Get the descriptive label for the given metric type (e.g. class accuracy, box accuracy, etc.)
  descriptor = get_metric_type_descriptor(metric_type)

  return metric_base_path + model + prefix + descriptor + ".csv"

#Writes a metric to file
#epoch - The epoch for this metric if applicable
#model_type - The type of model being run
#metric_type - The type of metric this metric entry is for
#cycle_type - Where in the cycle this is? Training? Validation? Testing? See CycleType enum.
#metric_val - The value to output
#iou - If applicable, the iou threshold for this metric. Different iou metrics save to different metric output files.
def write_metric(epoch, metric_type, cycle_type, metric_val, iou=-1.0):

  #Get the filename to write to
  metric_file_path = get_metric_file_path(metric_type, cycle_type)

  #Write to file
  write_file(epoch, str(metric_type) + "," + str(cycle_type) + "," + str(iou) + "," + str(metric_val), metric_file_path)

#Writes the log to file with the correct epoch.
#epoch - The current epoch if applicable. Otherwise set as -1
#logstr - The log string to write to file
def write_log(epoch, logstr):

  csv_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/logs/output.csv"   #LINUX
  #csv_path = "C:\\Phee\\Class\\Capstone\\logs\\output.csv"                              #WIN

  write_file(epoch, logstr, csv_path)
  print("epoch: " + str(epoch) + " ::: " + logstr)

#Segmentation callback - runs with semantic segmentation models
class EvaluateSegMetricsCallback(keras.callbacks.Callback):
  def __init__(self, data, save_path):
    super().__init__()
    self.data = data

    #Create a dictionary of metrics that we can iterate through later per validation record to calculate metrics.
    #Note: miou is used to save the model if the model achieves a miou higher than previous miou
    self.curr_metrics = {"pixel_accuracy" : keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy"), 
                    "miou" : keras.metrics.MeanIoU(num_classes=class_count, ignore_class=(0,), name="pixel_miou", sparse_y_true=True, sparse_y_pred=False)}
    
    self.save_path = save_path
    self.best_miou_val = -1.0

  #Fires when an epoch ends...
  def on_epoch_end(self, epoch, logs):

    #For each metric, refresh state
    for metric in self.curr_metrics.values():
      metric.reset_state()

    #For each image in the validation data, make pixel predictions and then evaluate each metric
    for batch in self.data:
      images, y_true = batch[0], batch[1]
      y_pred = self.model.predict(images, verbose=VERBOSE)
      for metric in self.curr_metrics.values():
        metric.update_state(y_true, y_pred)


    #TODO: FOR SEG, SAVE IMAGE RESULTS? THIS WOULD ALLOW TRAINING YOLO ON GENERATED SEG DATA RATHER THAN PREFABRICATED DATA, WHICH COULD HAVE BETTER RESULTS
    #if(TRAINING_SAVE_PROGRESSIVE_IMAGES):
    #  save_seg_masks(validation_path)

    #For each image and each metric, compile metrics log per image
    metric_log = {metric_name: float(metric.result()) for metric_name, metric in self.curr_metrics.items()}

    #Show metrics on log
    logs.update(metric_log)

    #https://docs.python.org/3/library/zoneinfo.html
    ts = time.time()
    logstr = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + " LOGS for epoch " + str(epoch) + " = " + str(logs).replace(",","")

    write_log(epoch, logstr)

    current_miou = metric_log["miou"]
    if current_miou > self.best_miou_val:
      self.best_miou_val = current_miou
      self.model.save(self.save_path)  # Save the model when mAP improves

    return super().on_epoch_end(epoch, logs)

  

#Keras/ TF callback info - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
#Run the evaluation for the YOLO model. This is only for instance segmentation (YOLO)
#Evaluates using COCO metrics - mAP
#This runs per step in the model I believe
class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    #Fires when an epoch ends. This allows for validation and updates the COCO metrics. It then reports them out to the log.
    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=VERBOSE)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)

        #Send COCO metrics mAP to the log
        logs.update(metrics)

        #https://docs.python.org/3/library/zoneinfo.html
        ts = time.time()
        logstr = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + " LOGS for epoch " + str(epoch) + " = " + str(logs).replace(",","")
        
        write_log(epoch, logstr)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs


# #The dice loss of the predicted data vs. the acutal, defined as 2*intersection divided by (sum(y_pred) + sum(y_actual))
# def get_dice_loss(y_pred, y_actual):

#TODO: WRITE INDIVIDUAL METRICS TO FILE - DURING TESTING AND/OR TRAINING/VALIDATION
#Write individual metrics to their own files based on the current epoch (or -1 if no epoch), cycle (training, validation, testing) as well
#epoch - The current epoch
#metric_type - Type of metric used (type of MetricType)
#cycle_type - What cycle is this in? Training? Validation? Testing?
def write_metric_file(epoch, metric_type, cycle_type, metric_val):
  metric_file_path = get_metric_file_path(metric_type, cycle_type)
  write_file(epoch, metric_val, metric_file_path)

#Given box x and y positioning values for predicted and actual, get the box intersection (used for iou and dice metrics)
#https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
#start_predx - The predicted bounding box starting x coordinate
#start_predy - The predicted bounding box starting y coordinate
#start_actx - The actual bounding box starting x coordinate
#start_acty - The actual bounding box starting y coordinate
#end_predx - The predicted bounding box ending x coordinate
#end_predy - The predicted bounding box ending y coordinate
#end actx - The actual bounding box ending x coordinate
#end_acty - The actual bounding box ending y coordinate
#returns the box intersection area between the predicted and actual box
def get_box_intersection(start_predx, start_predy, start_actx, start_acty, end_predx, end_predy, end_actx, end_acty):

  # determine the coordinates of the intersection rectangle
  x_left = max(start_predx, start_actx)
  y_top = max(start_predy, start_acty)
  x_right = min(end_predx, end_actx)
  y_bottom = min(end_predy, end_acty)

  if x_right < x_left or y_bottom < y_top:
      return 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)
  return intersection_area


#Can the given pixel navigate to the center of the given bounding box, using the mask as a nav mesh?
def can_nav_to_center_of_box():


#Can the given pixel navigate to at least one point on the given bounding box, using the mask as a nav mesh?
def can_nav_to_box():


#Gets the yolo bounding box that is matched to the given pixel coordinate. 
#x - The pixel x coordinate
#y - The pixel y coordinate
#yolo_boxes - The collection of yolo boxes here. Can be either predicted or actual boxes
def get_matched_box_to_pixel(x,y, yolo_boxes):


def determine_yoloseg_class():
  
def is_correct_object_detection(img_pred_classes, img_pred_boxes, img_actual_classes, img_actual_boxes)

  

#Get the given metric for the combined models (Yolo-Seg)
#yolo_y_pred - The YOLO y predicted values
#yolo_y_actual - The YOLO y actual values
#unet_y_pred - The unet y predicted values
#unet_y_actual - The unet y actual values
#metric_type - The desired metric type
#returns - The calculated metric determined by the provided metric_type
def get_yolo_seg_metrics(yolo_y_pred, yolo_y_actual, unet_y_pred, unet_y_actual, metric_type):

  #Make sure a predicted class is used for the unet prediction. Any class that is not nothing will just be "on" in this case
  unet_y_pred = np.argmax(unet_y_pred, -1, keepdims=True)

  #The predicted classes and boxes, where, each index is locked to the image index, and has an array of the classes and boxes, index-locked
  yolo_y_pred_classes = yolo_y_pred["classes"]
  yolo_y_pred_classes_np = keras.ops.convert_to_numpy(yolo_y_pred_classes)
  yolo_y_pred_boxes = yolo_y_pred["boxes"]
  yolo_y_pred_boxes_np = keras.ops.convert_to_numpy(yolo_y_pred_boxes)

  #Get the actual classes and boxes, which are packed into an array per index
  yolo_y_actual_classes = yolo_y_actual["classes"]
  yolo_y_actual_classes_np = keras.ops.convert_to_numpy(yolo_y_actual_classes)
  yolo_y_actual_boxes = yolo_y_actual["boxes"]
  yolo_y_actual_boxes_np = keras.ops.convert_to_numpy(yolo_y_actual_boxes)

  #Get actual pixel shape to march through in loop. Typically x and y dims will always be 512 and 512. Inst
  img_count = unet_y_actual.shape[0]
  shape_x = unet_y_actual.shape[1]
  shape_y = unet_y_actual.shape[2]

  #Convert dictionaries to simple numpy jagged arrays
  unet_y_pred_np = keras.ops.convert_to_numpy(unet_y_pred.to_numpy)
  unet_y_act_np = keras.ops.convert_to_numpy(unet_y_actual.to_numpy)

  #seg_metrics  - TP, FP, TN, FN for jsut segmentation (water vs. not water)
  true_positives = 0
  false_positives = 0
  true_negatives = 0
  false_negatives = 0

  #overall metrics - still mostly segmentation
  total_pixels = (img_count * shape_x * shape_y)
  total_true_water_pixels = 0
  total_pred_water_pixels = 0

  #Total number of yolo_seg pixels that have both a correct water pixel classification AND a correct yolo classification
  yolo_seg_pixels_correct = 0

  #Calculated ONLY on predicted water pixels. When water is predicted, the total number of yolo segmentation classifications that are actually correct
  #   This will help determine, if the segmentation is correct, if the YOLO object detection is also correct.
  #   Will also help determine which is more accurate - segmentation or object detection here
  yolo_seg_object_detections_correct = 0

  #The number of object detections only (only runs on predicted water pixels)
  yolo_seg_total_object_detections = 0

  #For each image in the image list
  for img_index in range(0, img_count):

    #Get the boxes for this image


    #For each x (horizontal) pixel
    for x in range(0, shape_x):

      #For each y (vertical) pixel
      for y in range(0, shape_y):

        #Get the actual pixel in this position. is it water?
        act_pixel = unet_y_act_np[img_index,x,y]
        is_actual_water = (act_pixel > 0)

        #Get the predicted pixel and actual pixel at this point (x,y) for an image...
        pred_pixel = unet_y_pred_np[img_index,x,y]
        is_pred_water = (pred_pixel > 0)

        #If the predicted pixel is water and the actual pixel is water, then see if a class can be assigned
        #Potential true positive / true positive for segmentation
        
        if(is_actual_water):
          total_true_water_pixels += 1
          if(is_pred_water):
            true_positives += 1
            total_pred_water_pixels += 1
          else:
            false_negatives += 1            

        #Water is not actually here in the ground truth images
        else:
          if(is_pred_water):
            false_positives += 1
            total_pred_water_pixels += 1
          else:
            true_negatives += 1

        #If water is predicted at all, then, run object detection on the water to get the class for it
        if(is_pred_water):

          #Gets the predicted classes and boxes for this image
          img_pred_classes = yolo_y_pred_classes_np[img_index]
          img_pred_boxes = yolo_y_pred_boxes_np[img_index]

          #Gets the actual classes and boxes for this image
          img_act_classes = yolo_y_actual_classes_np[img_index]
          img_act_boxes = yolo_y_actual_boxes_np[img_index]

            
          


          



#Gets the segmentation metrics
#y_pred - The predicted values from the model
#y_actual - The actual values from the model
#specific_class - If looking for a specific class
#metric_type - The type of metric to get
def get_seg_metrics_class(y_pred, y_actual, specific_class, metric_type):

  #Get the maximum prediction index (e.g. 0 if index 0 has the highest prediction of the <class count> predictions
  # Gets the predictions with the index of the BEST prediction in the final column (axis=-1 gets last column)
  y_pred = np.argmax(y_pred, -1, keepdims=True)

  #Get the: 0 - number of indices, 1 - height of image ground truth, 2 - width of image ground truth,
  instances = y_actual.shape[0]
  shape_x = y_actual.shape[1]
  shape_y = y_actual.shape[2]

  #Counters
  correct_predictions = 0
  true_positives = 0                #The index is X and the prediction is X (or for class, the index is X and the prediction is X)
  true_negatives = 0                #The index is 0 and the prediction is 0 (or for class, the index is Y and the prediction is Y)
  false_positives = 0               #The index is 0 and the prediction is X (or for class, the index is Y and the prediction is X)
  false_negatives = 0               #The index is X and the prediction is 0 (or for class, the index is X and the prediction is Y)
  correct_predictions = 0           #The number of correct predictions
  incorrect_predictions = 0         #The number of incorrect predictions
  classified_as_positive = 0        #Everything classified classified as positive
  all_predictions = 0               #This could be calculated from array shape, but, number of predictions. Need to count it bc of specific class counting predictions.

  #For each image index...
  for instance_idx in range(0, instances):

    print("Processing Image " + str(instance_idx))

    #Get the current actual image data (of [gt index],[x_val],[y_val],[pred_index])
    #Note: pred_index is not typical format. It's usually 3 predictions. We got the predicted index using argmax
    #   of the three predictions, which returned the index of the highest prediction 0-<class count>
    curr_y_act = y_actual[instance_idx]
    curr_y_pred = y_pred[instance_idx]        

    #For each x pixel index
    for pixel_idx_x in range(0, shape_x):

      #For each y pixel index
      for pixel_idx_y in range(0, shape_y):

        #Get the actual vs. predicted index
        index_actual = curr_y_act[pixel_idx_x][pixel_idx_y][0]
        index_pred = curr_y_pred[pixel_idx_x][pixel_idx_y][0]

        #Add to total predictions
        all_predictions += 1

        #If getting states for all classes, just have general TP,TN,FP,FN results
        if(specific_class == -1):

          #If classified, then, add to classified positives
          if(index_pred > 0):
              classified_as_positive += 1

          #If the pixel is wrong, record an incorrect prediction. Otherwise, record a correct prediction
          if(index_pred == index_actual):
            correct_predictions += 1
            if(index_actual == 0):
              true_negatives += 1
            else:
              true_positives += 1
          else:
            incorrect_predictions += 1
            if(index_actual == 0):
              false_positives += 1
            else:
              if(index_pred == 0):
                false_negatives += 1
        
        #Otherwise, comparison will be any other class than the current class 
        else:

          #True positive
          if(index_actual == specific_class and index_pred == specific_class):
            true_positives += 1
            correct_predictions += 1

          #True negative
          elif(index_actual != specific_class and index_pred != specific_class):
            true_negatives += 1
            correct_predictions += 1

          #False negative
          elif(index_actual == specific_class and index_pred != specific_class):
            false_negatives += 1
            incorrect_predictions += 1

          #False positive
          elif(index_actual != specific_class and index_pred == specific_class):
            false_positives += 1
            incorrect_predictions += 1

  #Get metrics
  accuracy = correct_predictions / all_predictions
  recall = true_positives / (true_positives + false_negatives)
  precision = true_positives / (true_positives + false_positives)
  f1_score = 2 * ((precision * recall) / (precision + recall))

  #Return the correct metric
  if(metric_type == MetricType.PIXELACCURACY):
    return accuracy
  elif(metric_type == MetricType.PIXELPRECISION):
    return precision
  elif(metric_type == MetricType.PIXELRECALL):
    return recall
  elif(metric_type == MetricType.PIXELDICE):
    return f1_score
  else:
    return 0.0



#Gets a class metric with a given class - this was originally created to just get IoU. Gets the IoU but also allows using it to get true positives for accuracy, precision, etc.
#y_pred - The prediction tensors by image index
#y_actual - The actual tensors by image index, index-locked with prediction tensors
#specific_class - The class to get metrics for
#metric_type - The metric type to use - see the MetricType enum
#iou_threshold - For metrics like accuracy, precision - the threshold for counting a positive or negative. Positive is >= threshold.
#returns - The requested metric
def get_yolo_metrics_class_box(y_pred, y_actual, specific_class, metric_type, iou_threshold):

  #The predicted classes and boxes, where, each index is locked to the image index, and has an array of the classes and boxes, index-locked
  y_pred_classes_per_image = y_pred["classes"]
  y_pred_classes_per_image_np = keras.ops.convert_to_numpy(y_pred_classes_per_image)
  y_pred_boxes_per_image = y_pred["boxes"]
  y_pred_boxes_per_image_np = keras.ops.convert_to_numpy(y_pred_boxes_per_image)

  #Get the actual classes and boxes, which are packed into an array per index
  y_actual_classes_per_image = y_actual["classes"]
  y_actual_classes_per_image_np = keras.ops.convert_to_numpy(y_actual_classes_per_image)
  y_actual_boxes_per_image = y_actual["boxes"]
  y_actual_boxes_per_image_np = keras.ops.convert_to_numpy(y_actual_boxes_per_image)

  #Get the number of ground truth classes
  img_count = y_pred_classes_per_image.shape[0]

  #The total iou to be divided for the average
  total_iou = 0.0

  #Total number of iou instances measured
  total_iou_instances = 0

  #Total classes in all images
  total_actual_classes = 0

  #Total predicted classes in all images
  total_predicted_classes = 0

  #Number of correct class classifications - incremented in loop below. used for accuracy, precision, and recall
  total_correct_class_classifications = 0

  #Initialize the counter of the number of positive box predictions based on the IoU threshold
  total_positive_box_ious = 0

  #Total number of predicted boxes in all images
  total_predicted_boxes = 0

  #Total actual boxes in all images
  total_actual_boxes = 0

  #Loop through each image array position, get the actual / predicted classes and boxes at this point and compare them
  for img_index in range(0, img_count):

    #Get the predicted and actual classes for this image index
    img_actual_classes = y_actual_classes_per_image_np[img_index]
    img_actual_boxes = y_actual_boxes_per_image_np[img_index]
    img_predicted_classes = y_pred_classes_per_image_np[img_index]
    img_pred_boxes = y_pred_boxes_per_image_np[img_index]

    #Filter classes for the passed in class or remove -1 padding - code below creates a true/false boolean array that acts as a mask to remove items from the array.
    if(specific_class > -1):

      #Create filter for specific class - removes padding as a result as well
      img_pred_class_filter = (img_predicted_classes == specific_class)
      img_actual_class_filter = (img_actual_classes == specific_class)
    else:
      #Create filters to filter out padding values (-1)
      #Similar to filtering with .any() below - https://www.w3schools.com/python/numpy/numpy_array_filter.asp - allows true/false masking to pull data subset
      img_pred_class_filter = (img_predicted_classes > -1)
      img_actual_class_filter = (img_actual_classes > -1)

    #Apply filter mask to classes and boxes
    y_pred_classes_filtered = img_predicted_classes[img_pred_class_filter]
    y_pred_boxes_filtered = img_pred_boxes[img_pred_class_filter]
    y_actual_classes_filtered = img_actual_classes[img_actual_class_filter]
    y_actual_boxes_filtered = img_actual_boxes[img_actual_class_filter]

    #https://stackoverflow.com/questions/51766102/converting-the-values-in-my-array-into-int - Also convert to int - bug where pred were float and act were int
    #Reassign the filtered arrays and convert them to numpy bc it's easier to work with numpy arrays imo than tensors which can be kind of annoying.
    img_predicted_classes = y_pred_classes_filtered
    img_predicted_classes_np = keras.ops.convert_to_numpy(img_predicted_classes).astype(int)
    img_pred_boxes = y_pred_boxes_filtered
    img_predicted_boxes_np = keras.ops.convert_to_numpy(img_pred_boxes)
    img_actual_classes = y_actual_classes_filtered
    img_actual_classes_np = keras.ops.convert_to_numpy(img_actual_classes).astype(int)
    img_actual_boxes = y_actual_boxes_filtered
    img_actual_boxes_np = keras.ops.convert_to_numpy(img_actual_boxes)

    #Get the number of predicted vs. actual classes
    predicted_class_count = img_predicted_classes_np.shape[0]
    total_predicted_classes += predicted_class_count
    predicted_box_count = img_predicted_boxes_np.shape[0]
    total_predicted_boxes += predicted_box_count
    actual_class_count = img_actual_classes_np.shape[0]
    total_actual_classes += actual_class_count
    actual_box_count = img_actual_boxes_np.shape[0]
    total_actual_boxes += actual_box_count

    #Get the number of times to loop - there could be too many or too few predictions
    number_of_loops = max(predicted_class_count, actual_class_count)   

    #Compute IOU based on the filtered predicted boxes and actual boxes
    #Returns an array of overlap IoUs per predicted box... Just take the maximum confidence to have the best IoU.
    curr_iou = keras.ops.convert_to_numpy(keras_cv.bounding_box.compute_iou(img_pred_boxes, img_actual_boxes,"xyxy"))

    #Get the number of positive IOU hits (if IOU count < number of loops, IoU will be penalized with zero for each extra loop)
    for iou_predictions in curr_iou:

      if(len(iou_predictions) > 0):
        #https://numpy.org/doc/2.4/reference/generated/numpy.argmax.html - Gets index
        #https://numpy.org/devdocs/reference/generated/numpy.max.html - Numpy array max value
        iou_val = iou_predictions.max()

        #If the current IoU is at or exceeds the IoU threshold, then, mark as positive result...
        if(iou_val >= iou_threshold):
          total_positive_box_ious += 1

        #Add IoU to the total IoU for averaging
        total_iou += iou_val
        total_iou_instances += 1

    #For either the complete number of predicted classe or the complete number of actual boxes) 
    for box_local_index in range(0, number_of_loops):

      #Initialize actual and predicted class, where -1 means it's invalid
      y_actual_class = -1
      y_predicted_class = -1

      #If still within bounds of the actual class count (because predicted box count can be > actual count) set the actual class
      if(box_local_index < actual_class_count):
        
        #Given the box local index (the index of a given box - actual, on ONE IMAGE - remember this is a sub loop inside a loop by image result)
        y_actual_class = img_actual_classes_np[box_local_index]

      #If still within bounds of the predicted class count (because actual box count can be > predicted box count) set the predicted class
      if(box_local_index < predicted_class_count):  
          
        #Given the box local index (the index of a given box - predicted, on ONE IMAGE - remember this is a sub loop inside a loop by image result)
        y_predicted_class = img_predicted_classes_np[box_local_index]

      if(y_actual_class > -1 and y_predicted_class > -1 and y_actual_class == y_predicted_class):
        total_correct_class_classifications += 1

  #If total predicted classes is zero, just return`` metric value as zero to avoid divide-by-zero error
  if(total_predicted_classes == 0 or total_positive_box_ious == 0 or total_actual_boxes == 0 or
     total_predicted_boxes == 0 or total_actual_classes == 0 or total_iou_instances == 0):
    return 0.0

  #Calculate metrics...
  class_accuracy = total_correct_class_classifications / total_predicted_classes
  box_accuracy = total_positive_box_ious / total_iou_instances
  class_recall = total_correct_class_classifications / total_actual_classes
  box_recall = total_positive_box_ious / total_actual_boxes
  class_precision = total_correct_class_classifications / total_predicted_classes
  box_precision = total_positive_box_ious / total_predicted_boxes

  #Special note for the metrics - Only UNet and DeepLab use True and False Negatives. These will not be factored into YOLO calculations, so, accuracy, precision,
  #     and recall, and, F1 score as a result, may be the same for YOLO. These are included for pixel measurements mostly and should be treated as 
  #     one metric for YOLO.

  #Get the total iou.
  # even though it would probably never happen with the current configuration.
  if(metric_type == MetricType.AVGIOU):
      return total_iou / total_iou_instances
  
  #Number of correct class predictions over all class predictions (count of predicted classes instead of ground truth)
  #(TP + TN) / (TP + TN + FP + FN)
  elif(metric_type == MetricType.CLASSACCURACY):
    return class_accuracy
  
  #Number of correct box prections for this IoU threshold over all box predictions (boxes predicted by the model, which may not match the count of GT boxes)
  #(TP + TN) / (TP + TN + FP + FN)
  elif(metric_type == MetricType.BOXACCURACY):
    return box_accuracy

  #TP / (TP + FN)
  elif(metric_type == MetricType.CLASSRECALL):
    return class_recall

  #TP / (TP + FN)
  elif(metric_type == MetricType.BOXRECALL):
    return box_recall

  #Number of class predictions that are actually correct 
  #TP / (TP + FP)
  elif(metric_type == MetricType.CLASSPRECISION):
      return class_precision

  #Number of correct box predictions out of all predicted box positive values (predictions made with IoU > threshold)
  #TP / (TP + FP)
  elif(metric_type == MetricType.BOXPRECISION):
    return box_precision

  #2 * ([Precision * recall] / [precision + recall])
  elif(metric_type == MetricType.CLASSF1):
     return 2 * ((class_precision * class_recall) / (class_precision + class_recall))

  elif(metric_type == MetricType.BOXDICE):
    return 2 * ((box_precision * box_recall) / (box_precision + box_recall))

  else:
     return 0.0

#Get the desired dataset (by its cycle type)
#data - The data to get the sub dataset from
#cycle-type - The current cycle data desired (training, validation, testing)
#all_img_count - The total number of images in the dataset
def get_dataset(data, cycle_type, all_img_count):

  #Splits off the test data and the train/val data as two distinct chunks. Then further splits train and validation data from the train/val chunk.
  #Get test data count
  #Special note: Training data is [TRAINING_DATA_PCT] of the image set data. Images will be
  # SHUFFLED every training data set by using random.shuffle(x)
  test_data_count = int(all_img_count * TEST_DATA_PCT)

  # Split the dataset into train and validation sets
  test_data = data.take(test_data_count)

  #Get the rest of the data that is not test data in this instance
  rest_of_data = data.skip(test_data_count)

  #Get validation size
  validation_size = int(len(rest_of_data) * VALIDATION_SPLIT_PCT)

  #If training, then
  val_data = rest_of_data.take(validation_size)
  train_data = rest_of_data.skip(validation_size)

  #Return the desired data
  if(cycle_type == CycleType.TRAINING):
    return train_data
  elif(cycle_type == CycleType.VALIDATION):
    return val_data
  else:
    return test_data


#Fetches data for training, validation, or testing cycles from their given datasets
#model_type - The type of model (YOLO, UNET) that is currently being used
#cycle_type - The current cycle (training, validation, testing) to get data for
#all_img_count - The total number of images in the dataset
def fetch_data(model_type, data, cycle_type, all_img_count):

  #Get the training, validation, or testing dataset
  ds = get_dataset(data, cycle_type, all_img_count)

  data_len = len(ds)
  len_ds_str = str(data_len)
  if(cycle_type == CycleType.TRAINING):
    print("Training Data Size = " + len_ds_str)
  elif(cycle_type == CycleType.VALIDATION):
    print("Validation Data Size = " + len_ds_str)
  else:
    print("Testing Data Size = " + len_ds_str)

  #Get augmentation steps
  #Get augmentation layers for Keras models
  augmenter = get_augmentation_layers(class_count, model_type)

  #UNet resizer. Gets replaced by yolo resizer if running yolo
  resizing = keras_cv.layers.Resizing(FILESIZE,FILESIZE,"nearest",pad_to_aspect_ratio=True)

  ####
  # Dataset Creation
  ####
  #Load training data
  #load_dataset - loads either the YOLO or UNet dataset based on the current model selected
  #https://keras.io/examples/vision/yolov8/
  if(model_type == ModelType.YOLO or model_type == ModelType.YOLO_SEG):

    ds = ds.map(load_yolo_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    #Function that resizes image to FILESIZExFILESIZE coordinates with nearest filtering. pad_to_aspect_ratio required for BB. BB format is xyxy
    resizing = keras_cv.layers.Resizing(FILESIZE,FILESIZE,"nearest",pad_to_aspect_ratio=True,bounding_box_format="xyxy")
    
  else:

      ds = ds.map(load_seg_data, num_parallel_calls=tf.data.AUTOTUNE)

  #Testing data does not get shuffled. Also, get full set of a ragged batch when testing
  if(cycle_type ==  CycleType.TRAINING or cycle_type == CycleType.VALIDATION):
    ds = ds.shuffle(BATCH_SIZE * 4)
    ds = ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
  else:
    ds = ds.ragged_batch(data_len, drop_remainder=False)

  #Only run augmentation on training images
  if(cycle_type == CycleType.TRAINING):

    #Only augment in training
    ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

  #If not training, them, only map resizer
  else:
    ds = ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE);

  #Based on model type, map function to the training dataset that converts to the data dictionary to a tuple format
  if(model_type == ModelType.YOLO or model_type == ModelType.YOLO_SEG):
    ds = ds.map(yolo_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
  else:
    ds = ds.map(seg_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

  #Prefetch the data for efficiency for all cycle types (Train, Validation, Testing)
  ds = ds.prefetch(tf.data.AUTOTUNE)
      
  return ds


 
#Get the data augumentation layers to use for models and return them as a set of augmentation steps
#file_size - The file size to resize images to
#model_type - The type of model being loaded. Valid selections are YOLO or UNET
def get_augmentation_layers(class_count, model_type):
  
  ####
  #Data Augmentation - Random Image Flips, Random Resizing
  ####
  #https://keras.io/examples/vision/yolov8/
  layers = []
  if(model_type == ModelType.YOLO):
    #layers.append(keras_cv.layers.Resizing(file_size,file_size,"nearest",pad_to_aspect_ratio=True,bounding_box_format="xyxy"))
    if(AUGMENTATION_FLIP_HORIZONTALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"))
    if(AUGMENTATION_FLIP_VERTICALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="vertical", bounding_box_format="xyxy"))
    if(AUGMENTATION_RANDOM_ROTATION):
      layers.append(keras_cv.layers.RandomRotation(factor=(0.0,1.0),bounding_box_format="xyxy", segmentation_classes=class_count))

    layers.append(keras_cv.layers.Resizing(FILESIZE,FILESIZE,"bilinear",pad_to_aspect_ratio=True,bounding_box_format="xyxy"))

  else:
    #layers.append(keras_cv.layers.Resizing(file_size,file_size,"nearest",pad_to_aspect_ratio=True))
    if(AUGMENTATION_FLIP_HORIZONTALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="horizontal"))
    if(AUGMENTATION_FLIP_VERTICALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="vertical"))
    if(AUGMENTATION_RANDOM_ROTATION):
      layers.append(keras_cv.layers.RandomRotation(factor=(0.0,1.0), segmentation_classes=class_count))

    layers.append(keras_cv.layers.Resizing(FILESIZE,FILESIZE,"nearest",pad_to_aspect_ratio=True))

  if(AUGMENTATION_RANDOM_GAUSSIAN_BLUR):
    layers.append(keras_cv.layers.RandomGaussianBlur(kernel_size=3,factor=(0.0,0.1)))
  if(AUGMENTATION_RANDOM_COLORJITTER):
    layers.append(keras_cv.layers.RandomColorJitter(value_range=(0,255),brightness_factor=(-0.05,0.05),contrast_factor=(0.0,0.05),saturation_factor=(0.35,0.45), hue_factor=(0.0,0.05)))


  augmenter = keras.Sequential(layers)
  return augmenter

#Output YOLO-SEG metrics to explore the viability of the entire model
def output_yolo_seg_metrics(yolo_y_pred, yolo_y_actual, unet_y_pred, unet_y_actual):



#Outputs the current unet metrics for a given class (or -1 for ALL. 0 is ignored because that's just background)
#y_pred - The Y Predicted values for UNET
#y_actual - The Y Actual values for UNET
#specific_class - The specific class to get metrics for. -1 Means ALL 
#curr_class_metric_type - The pixel metric used 
def output_unet_metrics_per_class(y_pred, y_actual, specific_class, curr_pixel_metric_type):
  step_str = "Overall "
  if(specific_class != -1):
    step_str = "Class [" + CLASS_LIST[specific_class + 1] + "]"   #Add 1 because unet has a "nothing" class as class 0, where as class 0 for YOLO is "Lake"
  pixel_metric = get_seg_metrics_class(y_pred, y_actual, specific_class, curr_pixel_metric_type)
  write_log(-1, step_str + get_metric_type_descriptor(curr_pixel_metric_type) + str(pixel_metric))
  write_metric(specific_class, curr_pixel_metric_type, CycleType.TESTING, pixel_metric)

#Outputs the current yolo metrics
#y_pred - The Y Predicted values for YOLO
#y_actual - The Y Actual values for YOLO
#specific_class - The specific class to get metrics for. -1 Means ALL
#curr_class_metric_type - The current CLASS metric type for YOLO
#curr_box_metric_type - The current BOX metric type for YOLO
def output_yolo_metrics_per_class(y_pred, y_actual, specific_class, curr_class_metric_type, curr_box_metric_type):

  #IoU Amouts to use as thresholds
  iou_thresholds = [0.5, 0.95]

  #For each IoU threshold, create a distinct report - only done for YOLO or YOLO seg
  for iou_threshold in iou_thresholds:
    class_metric = get_yolo_metrics_class_box(y_pred, y_actual, -1, curr_class_metric_type, iou_threshold)
    box_metric = get_yolo_metrics_class_box(y_pred, y_actual, -1, curr_box_metric_type, iou_threshold)
    step_str = "Overall "
    if(specific_class != -1):
      step_str = "Class [" + CLASS_LIST[specific_class] + "]"
    write_log(-1, step_str + get_metric_type_descriptor(curr_class_metric_type) + " [IoU=" + str(iou_threshold) + ")] = " + str(class_metric))
    write_log(-1, step_str + get_metric_type_descriptor(curr_box_metric_type) + "[IoU=" + str(iou_threshold) + ")] = " + str(box_metric))
    write_metric(specific_class, curr_class_metric_type, CycleType.TESTING, class_metric, iou_threshold)
    write_metric(specific_class, curr_box_metric_type, CycleType.TESTING, box_metric, iou_threshold)


#logs the actual precision and recall per class
#model_type - The type of model to make predictions for...
#y_pred - The y predicted values (from the model)
#y_actual - The actual ground truth values
#yolo_y_pred - The YOLO y predicted values if using yolo seg
#yolo_y_actual - The YOLO y actual values if using yolo seg
#unet_y_pred - The UNET y predicted values if using yolo seg
#unet_y_actual - The UNET y actual values if using yolo seg
def output_metrics_per_class(model_type, y_pred,y_actual, yolo_y_pred = None, yolo_y_actual = None, unet_y_pred = None, unet_y_actual = None):

  #Messages to write to the log / print out to the console
  log_messages = []

  #Classes to use. -1 is All, 0 is lakes, 1 is rivers
  class_indices = [-1, 0, 1]

  #Get the class and box metrics to be updated in the loop below (for general model and per class)
  class_metrics = [MetricType.CLASSACCURACY, MetricType.CLASSPRECISION, MetricType.CLASSRECALL, MetricType.CLASSF1]
  box_metrics = [MetricType.BOXACCURACY, MetricType.BOXPRECISION, MetricType.BOXRECALL, MetricType.BOXDICE]
  seg_metrics = [MetricType.PIXELACCURACY, MetricType.PIXELPRECISION, MetricType.PIXELRECALL, MetricType.PIXELDICE]

  #For each class index, starting with "All" and then going into specific classes,
  for class_index in class_indices:

    #For each metric type...
    for metric_index in range(0, len(class_metrics)):

      curr_class_metric_type = class_metrics[metric_index]
      curr_box_metric_type = box_metrics[metric_index]
      curr_pixel_metric_type = seg_metrics[metric_index]

      #If this is YOLO testing, then
      if(model_type == ModelType.YOLO):
        output_yolo_metrics_per_class(y_pred, y_actual, class_index, curr_class_metric_type, curr_box_metric_type)        
      #For UNET testing only
      elif(model_type == ModelType.UNET):
        curr_pixel_metric_type = seg_metrics[metric_index]
        output_unet_metrics_per_class(y_pred, y_actual, class_index, curr_pixel_metric_type)

      #For YOLO-SEG testing
      else:

        

#Given y_predicted values and y_actual values, calculate the IOU
#y_pred - The predicted values
#y_actual - The actual values
#specific_class - If this is for a specific class, the class number. -1 = all classes
def calculate_iou(y_pred, y_actual):
  pred_classes = y_pred["classes"]
  pred_boxes = y_pred["boxes"]
  actual_classes = y_actual["classes"]
  actual_boxes = y_actual["boxes"]

  return keras_cv.bounding_box.compute_iou(pred_boxes, actual_boxes, "xyxy")

#https://keras.io/examples/vision/oxford_pets_image_segmentation/
#***COPIED MODEL STRUCTURE FROM THIS SITE!***
#src:
  #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #: fchollet
  #DATE CREATED: 2019/03/20
  #DATE MODIFIED: 2020/04/20
  #***THIS SECTION OF THE CODE IS NOT MY OWN. IT WAS COPIED AND PASTED TO GET THE U-NET XCEPTION MODEL STRUCTURE.
  #Another note - I have modified some of the comments for clarity - Chris Albert
#Construct a u-net backbone to use for semantic segmentation
def get_unet_backbone():

  #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #Taken From this site ^^^^
  #CA - This is creating the Keras input tensor - especially the dimensions based on the file size, to be used by the model
  #CA - See: https://keras.io/api/layers/core_layers/input/
  inputs = keras.Input(shape=(FILESIZE,FILESIZE,3))

  ### [First half of the network: downsampling inputs] ### (credit: fchollet)

  #src: #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #src:
    #https://keras.io/examples/vision/oxford_pets_image_segmentation/
    #: fchollet
    #DATE CREATED: 2019/03/20
    #DATE MODIFIED: 2020/04/20
    #***THIS SECTION OF THE CODE IS NOT MY OWN. IT WAS COPIED AND PASTED TO GET THE U-NET XCEPTION MODEL STRUCTURE.
  # Entry block (credit: fchollet)
  x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #***COPIED MODEL STRUCTURE FROM THIS SITE!***
  #src:
    #https://keras.io/examples/vision/oxford_pets_image_segmentation/
    #: fchollet
    #DATE CREATED: 2019/03/20
    #DATE MODIFIED: 2020/04/20
    #***THIS SECTION OF THE CODE IS NOT MY OWN. IT WAS COPIED AND PASTED TO GET THE U-NET XCEPTION MODEL STRUCTURE.
  previous_block_activation = x  # Set aside residual

  #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #***COPIED MODEL STRUCTURE FROM THIS SITE!***
  #src:
    #https://keras.io/examples/vision/oxford_pets_image_segmentation/
    #: fchollet
    #DATE CREATED: 2019/03/20
    #DATE MODIFIED: 2020/04/20
    #***THIS SECTION OF THE CODE IS NOT MY OWN. IT WAS COPIED AND PASTED TO GET THE U-NET XCEPTION MODEL STRUCTURE.
  # Blocks 1, 2, 3 are identical apart from the feature depth. (credit: fchollet)
  for filters in [64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual (credit: fchollet)
    residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
      previous_block_activation
    )
    x = layers.add([x, residual])  # Add back residual (credit: fchollet)
    previous_block_activation = x  # Set aside next residual (credit: fchollet)

    ### [Second half of the network: upsampling inputs] ### (Credit: fchollet)

    #https://keras.io/examples/vision/oxford_pets_image_segmentation/
    #***COPIED MODEL STRUCTURE FROM THIS SITE!***
    #src:
      #https://keras.io/examples/vision/oxford_pets_image_segmentation/
      #: fchollet
      #DATE CREATED: 2019/03/20
      #DATE MODIFIED: 2020/04/20
      #***THIS SECTION OF THE CODE IS NOT MY OWN. IT WAS COPIED AND PASTED TO GET THE U-NET XCEPTION MODEL STRUCTURE.
  for filters in [256, 128, 64, 32]:
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(2)(x)

    # Project residual (credit: fchollet)
    residual = layers.UpSampling2D(2)(previous_block_activation)
    residual = layers.Conv2D(filters, 1, padding="same")(residual)
    x = layers.add([x, residual])  # Add back residual (credit: fchollet)
    previous_block_activation = x  # Set aside next residual (credit: fchollet)

  #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #***COPIED MODEL STRUCTURE FROM THIS SITE!***
  #src:
    #https://keras.io/examples/vision/oxford_pets_image_segmentation/
    #: fchollet
    #DATE CREATED: 2019/03/20
    #DATE MODIFIED: 2020/04/20
    #***THIS SECTION OF THE CODE IS NOT MY OWN. IT WAS COPIED AND PASTED TO GET THE U-NET XCEPTION MODEL STRUCTURE.
  # Add a per-pixel classification layer (credit: fchollet)
    outputs = layers.Conv2D(class_count, 3, activation="softmax", padding="same")(x)

  #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #***COPIED MODEL STRUCTURE FROM THIS SITE!***
  #src:
    #https://keras.io/examples/vision/oxford_pets_image_segmentation/
    #: fchollet
    #DATE CREATED: 2019/03/20
    #DATE MODIFIED: 2020/04/20
    #***THIS SECTION OF THE CODE IS NOT MY OWN. IT WAS COPIED AND PASTED TO GET THE U-NET XCEPTION MODEL STRUCTURE.
  # Define the model (credit: fchollet)
  model = keras.Model(inputs, outputs)
  return model






#https://keras.io/examples/vision/oxford_pets_image_segmentation/
#Perform segmentation model training, given a training dataset and validation dataset
#train_ds - The training dataset
#val_ds - The validation dataset (validated during training)
def seg_train_model(train_ds, val_ds):

  #Construct the u-net backbone
  seg_model = get_unet_backbone()

  #Also built from https://deeplearningwithpython.io/chapters/chapter11_image-segmentation/ as a src
  #What do these do?
    #SparseCategoricalAccuracy - This is pixel accuracy (accuracy per pixel on the images)
    #MeanIoU - This is the mean IOU of the segmentation masks. Example uses the foreground but I'll likely use the classes themselves

  #Set up training metrics
  metrics = [keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy"), 
                    keras.metrics.MeanIoU(num_classes=class_count, name="pixel_miou", sparse_y_true=True, sparse_y_pred=False)]

  #Choose the optimizer
  #https://keras.io/examples/vision/yolov8/
  if(OPTIMIZER == Optimizer.ADAM):
    optimizer = tf.keras.optimizers.Adam(
      learning_rate=LEARNING_RATE,
      global_clipnorm=GLOBAL_CLIPNORM,
      weight_decay = WEIGHT_DECAY,
      epsilon=1e-07,
    )
  elif(OPTIMIZER == Optimizer.RMSPROP):
    optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=LEARNING_RATE,
      global_clipnorm=GLOBAL_CLIPNORM,
      weight_decay = WEIGHT_DECAY,
      epsilon=1e-07,
    ) 
  
  #Compile the model
  seg_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=metrics)

  #https://keras.io/examples/vision/oxford_pets_image_segmentation/

  #https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU
  #Create model checkpoint to save weights
  #What do these do?
    #ModelCheckpoint - Saves the weights from the model when running. This currently not used, but, might be later. Only saves weights if the metrics improve
    #EvaluateSegMetricsCallback - Evaluates the segmentation model with metrics such as pixel accuracy
  model_path = get_path(PathType.MODEL, ModelType.UNET)
  seg_checkpoint_path = get_path(PathType.CHECKPOINT, ModelType.UNET)
  callbacks = [keras.callbacks.ModelCheckpoint(filepath=seg_checkpoint_path, save_best_only=True), EvaluateSegMetricsCallback(data=val_ds, save_path=model_path)]

  #https://keras.io/examples/vision/oxford_pets_image_segmentation/
  #Peform Training!
  seg_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
  )
  

#Perform yolo training, given a training dataset and validation dataset
#train_ds - The training dataset
#val_ds - The validation dataset (validated during training)
def yolo_train_model(train_ds, val_ds):

  #https://keras.io/examples/vision/yolov8/
  Model_Size_String = "yolo_v8_m_backbone_coco"
  if(MODEL_SIZE == ModelSize.Small):
    Model_Size_String = "yolo_v8_s_backbone_coco"
  elif (MODEL_SIZE == ModelSize.Large):
    Model_Size_String = "yolo_v8_l_backbone_coco"
  elif (MODEL_SIZE == ModelSize.XtraSmall):
    Model_Size_String = "yolo_v8_xs_backbone_coco"
  elif (MODEL_SIZE == ModelSize.XtraLarge):
    Model_Size_String = "yolo_v8_xl_backbone_coco"

  # Set backbone
  #https://keras.io/examples/vision/yolov8/
  backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    Model_Size_String 
  )

    #https://keras.io/examples/vision/yolov8/
  yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=2,    #Changed from 1 to 2
  )

  #Choose the optimizer
  #https://keras.io/examples/vision/yolov8/
  if(OPTIMIZER == Optimizer.ADAM):
    optimizer = tf.keras.optimizers.Adam(
      learning_rate=LEARNING_RATE,
      global_clipnorm=GLOBAL_CLIPNORM,
      weight_decay = WEIGHT_DECAY,
      epsilon=1e-07,
    )
  elif(OPTIMIZER == Optimizer.RMSPROP):
    optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=LEARNING_RATE,
      global_clipnorm=GLOBAL_CLIPNORM,
      weight_decay = WEIGHT_DECAY,
      epsilon=1e-07,
    ) 

  #Compile the model
  #https://keras.io/examples/vision/yolov8/
  yolo.compile(
      optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
  )

  #Run the training loop for the model
    #This also saves the model as a file that can be loaded later for validation...
  #https://keras.io/examples/vision/yolov8/
  #https://github.com/keras-team/keras-cv/issues/2095 (as a code reference)
  #Precision and recall: https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras
  model_path = get_path(PathType.MODEL, ModelType.YOLO)
  yolo.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS,
      callbacks=[EvaluateCOCOMetricsCallback(val_ds, model_path), keras_cv.callbacks.PyCOCOCallback(validation_data=val_ds, bounding_box_format="xyxy")],
      verbose=VERBOSE,
  )


#Visualize the incoming training YOLO data to make sure it has accurate ground truths
def visualize_training_data(train_ds, bounding_box_format):
  images, y_actual = next(iter(train_ds.take(1)))
  visualization.plot_bounding_box_gallery(
    images,
    value_range=(0, 255),
    bounding_box_format=bounding_box_format,
    y_true=y_actual,
    scale=1,
    rows=10,
    cols=10,
    font_scale=1,
    class_mapping=class_mapping,
  )  


#Given the class name, convert it to the class index
#classname (string) - The name of the class to convert to an index - must match values in CLASS_LIST
def getClassIndexFromName(classname) -> int:
  for i in range(0,len(CLASS_LIST)):
    current_class_name = CLASS_LIST[i]
    if(current_class_name == classname):
      return i
  
  #Class name has not been found
  return -1

#Given the class name, convert it to the class index
#classname (string) - The name of the class to convert to an index - must match values in CLASS_LIST
def getClassIndexFromName(classname) -> int:
  for i in range(0,len(CLASS_LIST)):
    current_class_name = CLASS_LIST[i]
    if(current_class_name == classname):
      return i
  #
  #Class name has not been found
  return -1

#Saves a segmentation image with accumulated pixels and the base image path
#base_image_path - The base image path that will be used to save the segmentation image
#img_pixels - The pixels to save for this image
#returns - The path that this segmentation image was saved to.
def save_seg_img(base_image_path, img_pixels):
    
    #https://www.geeksforgeeks.org/python/python-pil-image-new-method/
    #Create new image
    new_img = Image.new("RGB", (FILESIZE,FILESIZE))
    new_img.putdata(img_pixels)

    #https://www.w3schools.com/python/ref_string_replace.asp
    #Update image path to be a mask output path
    mask_img_path = base_image_path.replace("seg_images","output_masks")

    #Saves the image
    new_img.save(mask_img_path)

    return mask_img_path


#Given the segmentation mask data, save those masks to the output directory
#image_paths - The collection of image paths passed into the model for training
#y_pred - The Y predicted values
def save_seg_masks(image_paths, y_pred):

    #TODO: I THINK THIS IS SAVING IMAGES TO THE INCORRECT NUMBERS...
    #CHECK TRAINING FOR THAT AS WELL...
    
    #Seg image paths that were saved. Will be returned from this model and used to make predictions...
    seg_img_paths = []

    y_pred = np.argmax(y_pred, axis=-1, keepdims=True)

    #The number of images in this image set
    image_count = y_pred.shape[0]

    #The size of the x dimension per image
    x_size = y_pred.shape[1]

    #The size of the y dimension per image
    y_size = y_pred.shape[2]
    
    #Current image pixels to save
    img_pixels = []

    #The previous image index. When doesn't equal current image index, will save img data and dump pixels for next image
    prev_img_index = 0

    #For image index 0 to <image_count> - gets the first 3 dimensions of the array and gets the values from those. Loops through each - a,b,c syntax for triple loop. ndindex is triple-nested loop replacement.
    for imgindex, x, y in np.ndindex(y_pred.shape[:3]):

        #Get the current predicted pixel (index 3) from the predictions - the predicted pixel is the calculated max index from the predictions
        predicted_pixel = y_pred[imgindex, x, y]

        #Gets mask and outputs as binary mask for white where water exists vs. black for where it does not
        pval = max(0,min(predicted_pixel[0],1)) * 255
        rgb_value = (pval,pval,pval)

        #Add pixel to image pixels
        img_pixels.append(rgb_value)

        #Sets the previous image index to this index for next loop
        prev_img_index = imgindex

        #If the current index doesn't match the previous index, save the image to file
        if(x == x_size - 1 and y == y_size - 1):

          #Saves the current segmentation image based on accumulated pixels
          img_save_path = image_paths[imgindex]
          saved_seg_img_path = save_seg_img(img_save_path, img_pixels)

          #Add the saved segmentation image path to the list of saved segmentation images
          seg_img_paths.append(saved_seg_img_path)

          #Purge currently saved pixels to start writing for new image...
          img_pixels = []


def get_predicted_data(model_type, dataset):
  if(model_type == ModelType.YOLO):
    #https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
    yolo_model_path = get_path(PathType.MODEL, ModelType.YOLO)
    yolo_model = keras.saving.load_model(yolo_model_path)
  elif(model_type == ModelType.UNET):      
    #https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
    unet_model_path = get_path(PathType.MODEL, ModelType.UNET)
    unet_model = keras.saving.load_model(unet_model_path)

  #Load the images and either classes / bounding boxes, or prediction images
  #These are not from the model. They're taken from file, and, same data used for both YOLO and unet
  print("Processing Testing Images...")
  images, y_actual = next(iter(dataset.take(1)))

  #Get predicted
  yolo_prediction_count = 0
  segmentation_prediction_count = 0
  if(model_type == ModelType.YOLO):
    write_log(-1, "Running YOLO Model Prediction...")
    yolo_y_pred = yolo_model.predict(images, batch_size=BATCH_SIZE)
    return yolo_y_pred

  else:
    write_log(1, "Running UNet Model Prediction...")
    seg_y_pred = unet_model.predict(images, batch_size=BATCH_SIZE)
    return seg_y_pred


#Run testing cycle after training and validation
#https://keras.io/examples/vision/yolov8/
#model_type - The type of model used
#data - The loaded dataset
#cycle_type - The cycle of data to do this testing for (training, validation, testing - usually testing unless generating predictions...)
#bounding_box_format - For YOLO, the bounding box format. Usually xyxy
#all_img_count - The number of images in the dataset...
#no_metrics - If true, will not show calculate or show metrics
#yolo_dataset - Optional - If using yolo seg, the YOLO dataset
#set_dataset - Optional - If using yolo seg, the segmentation dataset
def do_testing(model_type, data, cycle_type, bounding_box_format, all_img_count, no_metrics, yolo_dataset = None, seg_dataset = None):


  #YOLO-Seg gets predictions for both YOLO and UNet Segmentation. The other two run independently for testing each.
  if(model_type == ModelType.YOLO_SEG):

    #Get YOLO image paths
    yolo_img_path_data = get_dataset(yolo_dataset, cycle_type, all_img_count)
    yolo_img_paths = get_paths_from_dataset(yolo_img_path_data, ModelType.YOLO, ImageDataType.RGB)

    #Get Seg image paths
    seg_img_path_data = get_dataset(seg_dataset, cycle_type, all_img_count)
    seg_img_paths = get_paths_from_dataset(seg_img_path_data, ModelType.UNET, ImageDataType.RGB)


    #Get the YOLO data to use for making predictions...
    yolo_ds = fetch_data(ModelType.YOLO, yolo_dataset, cycle_type, all_img_count)

    #Get predictions for yolo
    yolo_y_pred = get_predicted_data(ModelType.YOLO, yolo_ds)

    #Load the images and either classes / bounding boxes, or prediction images
    #These are not from the model. They're taken from file, and, same data used for both YOLO and unet
    print("Processing YOLO Testing Images...")
    yolo_images, yolo_y_actual = next(iter(yolo_ds.take(1)))

    #Get the UNET data to use for making predictions...
    unet_ds = fetch_data(ModelType.UNET, seg_dataset, cycle_type, all_img_count)

    #Load the images and either classes / bounding boxes, or prediction images
    #These are not from the model. They're taken from file, and, same data used for both YOLO and unet
    print("Processing UNET Testing Images...")
    unet_images, unet_y_actual = next(iter(unet_ds.take(1)))

    #Get predictions for UNET
    unet_y_pred = get_predicted_data(ModelType.UNET, unet_ds)
    
  else:

    #Fetch data for the requested cycle type
    #Get test image paths - get test data selection so that we can get the paths from it
    img_path_data = get_dataset(data, cycle_type, all_img_count)
    img_paths = get_paths_from_dataset(img_path_data, model_type, ImageDataType.RGB)

    #Get the datasource to use for making predictions...
    ds = fetch_data(model_type, data, cycle_type, all_img_count)

    #Split simple dataset into images and y_actual
    images, y_actual = next(iter(ds.take(1)))

    #Get the predicted data for the given model type if it's for individual yolo or unet
    y_pred = get_predicted_data(model_type, ds)

  #Gets the the total segmentation and yolo prediction count - 
  # If individual, only one will have values
  # If YOLO seg, both will have values
  segmentation_prediction_count = 0
  yolo_prediction_count = 0 
  if(model_type == ModelType.YOLO):
    yolo_prediction_count = y_pred["boxes"].shape[0]
  elif(model_type == ModelType.UNET):
    segmentation_prediction_count = y_pred.shape[0]
  else:
    yolo_prediction_count = yolo_y_pred["boxes"].shape[0]
    segmentation_prediction_count = unet_y_pred.shape[0]

  #If this is a segmentation test save the segmentation images so that we can run YOLO on it next for the classification...
  if(model_type == ModelType.UNET):
      #If saving segmentation images is enabled, then
      if(OPTION_SAVE_SEG_IMAGES):
          #Save the segmentation images
          save_seg_masks(img_paths, unet_y_pred)

  #IF Yolo or UNET only, one of these will be zero. Will still report the correct number of predictions...
  total_pred =  yolo_prediction_count + segmentation_prediction_count

  #Output count of boxes
  print("Prediction Count = " + str(total_pred))
  print("YOLO Prediction Count = " + str(yolo_prediction_count))
  print("UNET Prediction Count = " + str(segmentation_prediction_count))

  if(model_type == ModelType.YOLO_SEG):
    yolo_image_count = len(yolo_images)
    seg_image_count = len(unet_images)
    total_image_count = yolo_image_count + seg_image_count
    print("Total Images: " + str(total_image_count))
    print("YOLO Images: " + str(yolo_image_count))
    print("UNET Images: " + str(seg_image_count))
  else:
    #Get and display the number of images
    num_images = len(images)
    print("There are " + str(num_images) + " images in this dataset...")

  #If true, will calculate and show the metrics. If false, will do neither and skip this step all together
  if(OPTION_CALCULATE_METRICS and no_metrics==False):

    #https://zihaogeng.medium.com/how-to-evaluate-an-object-detection-model-iou-precision-recall-and-map-f7cc12e0dcf6
    output_metrics_per_class(model_type, yolo_y_pred, y_actual)   #YOLO Metrics Out
      
  #If we are showing the output images then...
  if(OPTION_SHOW_OUTPUT_IMAGES and no_metrics==False):

      #If this model type is yolo, then, show yolo images
      if(model_type == ModelType.YOLO):

        #Visualization for YOLO
        visualization.plot_bounding_box_gallery(
            images,
            value_range=(0, 255),
            bounding_box_format=bounding_box_format,
            y_true=y_actual,
            y_pred=y_pred,
            scale=1,
            rows=5,
            cols=5,
            font_scale=1,
            class_mapping=class_mapping,
        )

      #If this is image segmentation, then...
      elif(model_type == ModelType.UNET):

        #Get the predictions with the index of the BEST prediction in the final column (axis=-1 gets last column)
        y_pred_max = np.argmax(y_pred, axis=-1, keepdims=True)

        #print("y_pred_max_shape = "  + str(y_pred_max.shape[0]) + "," + str(y_pred_max.shape[1]) + "," + str(y_pred_max.shape[2]) + "," + str(y_pred_max.shape[3]))

        #Visualization for semantic segmentation
        visualization.plot_segmentation_mask_gallery(
          images=images,
          y_pred=y_pred_max,
          y_true=y_actual,
          value_range=(0,255),
          num_classes=class_count,
          rows=5,
          cols=5
        )


#Given a dataset, loops through and gets the number of paths
#dataset - The dataset to get the paths from
#image_data_type - The type of image (ImageDataType) to get from the dataset - RGB data? Masks?'
#model_type - The type of model to get data for. YOLO and UNet have different data structures. YOLO is [ImgPath, Class, BBox] and UNET is [ImgPath, MaskPath]
#returns - A list of paths from the given dataset for the given data type
def get_paths_from_dataset(dataset, model_type, image_data_type):

    #Get image and mask (if desired) paths
    test_img_paths = []
    test_mask_paths = []

    if(model_type == ModelType.UNET):

      #Loop through [ImgPath, MaskPath] dictionary
      for imgpath, maskpath in dataset:

        #For UNet, get the image and mask paths
        test_img_paths.append(imgpath.numpy().decode("utf-8"))
        test_mask_paths.append(maskpath.numpy().decode("utf-8"))
    
    #Get YOLO image paths only
    else:

        #Loop through [ImgPath, Classes, BoundingBoxes] dictionary
        for imgpath, classes, bbox in dataset:

          #Add the YOLO Image path to the list of image paths
          test_img_paths.append(imgpath.numpy().decode("utf-8"))

    #Based on the desired image data type, return the list of paths
    if(image_data_type == ImageDataType.RGB):
      return test_img_paths
    else:
      return test_mask_paths

    
#Prepares the YOLO dataset that will be split into training, validation, test  data
def prepare_yolo_data(mask_full_paths):

  #PROBLEM - IMAGE SEG FILE NAMES AND MASK FILE NAMES DO NOT MATCH THE ANNOTATION FILE NAMES!
  #THEREFORE, THESE CANNOT BE REFERENCED FROM ANNOTATIONS
  #ANNOTATIONS FILE NEEDS TO BE READ DIFFERENTLY - MAYBE JUST THE FIRST PART OF IT? NEED TO IGNORE THE SECOND PART
  # WHERE THERE IS A GUID. THE GUID OF SEG IMAGES DOESNT MATCH THE GUID OF THE YOLO IMAGES

  #ALSO TODO: MIGHT NOT BE LOADING CORRECT BOUNDING BOXES

  #LOGIC - 
  #1 - SPLIT THE ANNOTATION FILENAME VALUE BY .
  #2 - USE THE FIRST ELEM FROM THIS ANNOTATION VALUE AND REMOVE _JPG AND _PNG TO SEARCH AGAINST img_full_paths - GET THE FIRST PART OF img_full_paths TOO WITHOUT _JPG OR _PNG
  #3 - WHEN FOUND, USE THOSE BOUNDING BOXES 

  # Read the entire content of the bounding box file
  bounding_box_annotations_path = "/home/chrissy/Documents/School/CS547/AIStuff/TestYOLO/_annotations.csv"   #LINUX
  #bounding_box_annotations_path = "C:\\Phee\\Class\\Capstone\\img\\yolo\\_annotations.csv"    #WIN
  
  lines = read_text_file(bounding_box_annotations_path)

  #Create temp class list
  classes = []

  bounding_boxes = []

  filenames = []

  #Populate bounding boxes array and class array per image file list item
  for img_index in range(0,len(mask_full_paths)):
    bounding_boxes.append([])
    #classes.append([0])
    classes.append([])
    
  #For each bounding box found, add the bounding box for this image
  for i in range(0, len(lines)):

    line = lines[i]
    if(not line.startswith("filename")):
      vals = line.split(",")      
      filename = vals[0]

      #Get the first part of the filename before the GUID to search for the correct matched image to this one
      file_parts = filename.split('.')

      #Get the unique file descriptor that is not a guid or file extension to match seg images with yolo images
      #Include the slash in front as to make sure is a starts with check for the file name
      
      
      actual_file_descriptor = "/" + file_parts[0]       #LINUX - REMEMBER TO CHANGE THIS BACK!
      #actual_file_descriptor = "\\" + file_parts[0]       #WIN

      #Get the class name
      classname = vals[3]

      #Get bounding box class index from its name
      class_index = getClassIndexFromName(classname)

      #Get image index from image list
      img_index = -1

      #Get image index from filename
      #https://www.geeksforgeeks.org/python/how-to-find-the-index-of-value-in-numpy-array/
      #img_index = np.where(img_full_paths == filename)[0]
      #This image is used to add bounding boxes to this image, not to get the path
      img_index = -1
      for i in range(0,len(mask_full_paths)):
        img_path = mask_full_paths[i]

        #Does the file descriptor exist in this image path? If so, this is the correct mask image to use for this YOLO entry...
        if((actual_file_descriptor in img_path) and img_index == -1):

          img_index = i
          
          x0 =  float(vals[4])
          y0 =  float(vals[5])

          x1 =  float(vals[6])
          y1 =  float(vals[7])

          #Create formatted bounding box
          bbox = BoundingBox(img_index, class_index, x0, y0, x1, y1)

          #filenames.append(img_path)

          #Add Bounding Box
          bounding_boxes[img_index].append(bbox.toTensor())
          classes[img_index].append(class_index)
          #print("Appended class " + str(class_index))
          #classes[img_index].append(1)

          #If should report on matched YOLO to bounding boxes, do this
          if(REPORT_YOLO_FILE_MATCHES):
            yolo_bb_log = img_path + "," + str(class_index) + "," + str(x0) + "," + str(y0) + "," + str(x1) + "," + str(y1) + "idx=" +  str(img_index) + "line=" + line
            write_log("OUTPUT_YOLO_LOGS", yolo_bb_log)

      if(img_index == -1):
        write_log("ERROR", "Mask Img Not Found - Desc=" + actual_file_descriptor)


  #https://keras.io/examples/vision/yolov8/
  write_log(-1, "Structuring Data...")
  image_paths = tf.ragged.constant(mask_full_paths)
  bbox = tf.ragged.constant(bounding_boxes)
  classes = tf.ragged.constant(classes)

  #Create dictionary from tensor "slices" - e.g. image paths, classes bounding boxes
  data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
      
  #returns the data dictionary that gets split into training, test, etc. data
  return data


#Prepares data for semantic segmentation, which will be split into training, validation, test datasets
def prepare_seg_data(img_full_paths, mask_full_paths):

  image_paths = tf.ragged.constant(img_full_paths)
  mask_paths = tf.ragged.constant(mask_full_paths)
  data = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
  return data

#Gets the given path type based on parameters...
#path_type - The desired type of path to get, e.g. for images or model
#model_type - The type of model being run
#yolo_training_source - The source of images being trained on or viewed for YOLO, e.g. original images, prefab masks from roboflow, or, mask predictions from the unet model
#returns - The model path to load the model from or save the model to...
def get_path(path_type, model_type, yolo_training_source = YOLOTrainingSource.OriginalImages):

  #Gets the model checkpoint path for image segmentation
  if(path_type == PathType.CHECKPOINT):
    return "/home/chrissy/Documents/School/CS547/AIStuff/capstone/models/seg_chk.keras"  #LINUX
    #return "C:\\Phee\\Class\\Capstone\\seg_chk.keras"                                     #WIN

  #Set the filepath (for saving and loading)
  yolo_model_path = "/home/chrissy/Documents/School/CS547/AIStuff/TestYOLO/model.keras"        #LINUX
  unet_model_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/models/unet.keras"  #LINUX

  #yolo_model_path = "C:\\Phee\\Class\\Capstone\\model.keras"                                    #WIN
  #unet_model_path = "C:\\Phee\\Class\\Capstone\\unet.keras"                                     #WIN

  #Get correct image paths...
  if(yolo_training_source == YOLOTrainingSource.PrefabMasks or yolo_training_source == YOLOTrainingSource.OriginalImages):
    yolo_img_path ="/home/chrissy/Documents/School/CS547/AIStuff/capstone/seg_images/"   #LINUX
    #yolo_img_path ="C:\\Phee\\Class\\Capstone\\img\\yolo\\"                              #WIN
  else:
    yolo_img_path ="/home/chrissy/Documents/School/CS547/AIStuff/capstone/output_masks/" #LINUX
    #yolo_img_path = "C:\\Phee\\Class\\Capstone\\img\\yolo\\"                              #WIN
  seg_img_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/seg_images/"     #LINUX
  #seg_img_path = "C:\\Phee\\Class\\Capstone\\img\\seg\\"                                  #WIN

  #Set the paths per model to return...
  if(model_type == ModelType.YOLO):
    model_path = yolo_model_path
    image_data_path = yolo_img_path
  else:
    model_path = unet_model_path
    image_data_path = seg_img_path

  #Return the desired data
  if(path_type == PathType.MODEL):
    return model_path
  else:
    return image_data_path


#Given a model type and directory path, get all files in that directory
def get_dir_data(model_type, dir_path):
  
  #Only sort if generating unet images
  if(model_type == ModelType.UNET):
    dirdata = sorted(os.listdir(dir_path))
  else:
    dirdata = os.listdir(dir_path)

  return dirdata


#Prepares data to be processed for the given model type...
#model_type - The data type to prepare the data for
#yolo_training_source - If yolo, the data source to train off of. Ignored if unet
#returns - The prepared data of image paths,classes per image path, and bounding boxes per path  if yolo, or, image path and mask path for unet
def prepare_data(model_type, yolo_training_source=YOLOTrainingSource.OriginalImages):

  #Load the paths for masks and rgb training images
  mask_full_paths = get_image_file_paths(model_type, ImageDataType.MASKS)
  img_full_paths = get_image_file_paths(model_type, ImageDataType.RGB)

  #Get dataset to use based on the current model type YOLO, U-NET, etc.
  if(model_type == ModelType.YOLO or model_type == ModelType.YOLO_SEG):
    if(yolo_training_source == YOLOTrainingSource.PrefabMasks):
      data = prepare_yolo_data(mask_full_paths)
    else:
      data = prepare_yolo_data(img_full_paths)
  else:
    data = prepare_seg_data(img_full_paths, mask_full_paths)

  return data
  

#Gets image paths from a directory.
#model_type - The current model type
#dir_path - The directory path to load images from
#image_data_type - The image data type being loaded (e.g. rgb, masks)
def get_image_file_paths(model_type, image_data_type):

  #The path to current images to load
  dir_path = get_path(PathType.IMAGES, model_type)

  #Get  directory data
  dirdata = get_dir_data(model_type, dir_path)

  final_paths = []

  fileext = ".jpg"
  if(image_data_type == ImageDataType.MASKS):
    fileext = ".png"
    
  #Select mask files from all files
  for genericFile in dirdata:
    name, fileExtension = os.path.splitext(genericFile)
    if fileExtension == fileext:
      final_paths.append(dir_path + genericFile)

  return final_paths


#Check if the settings are being run in a valid state - e.g., can't running training on YOLO_SEG because that's only for testing.
#returns - True if state is valid, or, false if state is invalid
def check_for_valid_state():

    #Invalid state - running training when YOLO_SEG is selected. To train for YOLO, just select ModelType.YOLO instead...
    if(MODE == Mode.TRAIN and MODEL_TYPE == ModelType.YOLO_SEG):
        return False

    #Did not hit any invalid states
    return True

#Print the starting message
def start_message():
  #Put start message in logs
  ts = time.time()
  logstr = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + " --- STARTED!"
  write_log(-1, logstr)

#Start process - either trains or tests
def start(model_type):

  #-----------------------------------#
  # Entry Point - Where the code actually starts doing stuff...
  #-----------------------------------#

  #Print the script starting message
  start_message()

  #Check for valid state and end if not valid
  is_valid_state = check_for_valid_state()

  #If running in invalid state, then
  if(is_valid_state == False):
      write_log("ERROR", "ERROR - INVALID STATE DETECTED! PLEASE CHECK SETTINGS (MODE, MODEL TYPE, ETC) AND TRY AGAIN.")
      return None

  #Load the paths for masks and rgb training images
  mask_full_paths = get_image_file_paths(model_type, ImageDataType.MASKS)
  img_full_paths = get_image_file_paths(model_type, ImageDataType.RGB)

  #Show full image count and full mask count...
  all_img_count = len(img_full_paths)
  all_mask_count = len(mask_full_paths)
  write_log(-1, "Total Images = " + str(all_img_count))
  write_log(-1, "Total Masks = " + str(all_mask_count))

  if(model_type == ModelType.YOLO_SEG):
    yolo_data = prepare_data(ModelType.YOLO)
    seg_data = prepare_data(ModelType.UNET)

    #Get yolo training and validation data
    # yolo_train_ds = fetch_data(model_type, yolo_data, CycleType.TRAINING, all_img_count)
    # yolo_val_ds = fetch_data(model_type, yolo_data, CycleType.VALIDATION, all_img_count)

    # #Get segmentation training and validation data
    # seg_train_ds = fetch_data(model_type, seg_data, CycleType.TRAINING, all_img_count)
    # seg_val_ds = fetch_data(model_type, seg_data, CycleType.VALIDATION, all_img_count)

    
  else:
    data = prepare_data(model_type)

    #Fetch training, validation, and testing data
    train_ds = fetch_data(model_type, data, CycleType.TRAINING, all_img_count)
    val_ds = fetch_data(model_type, data, CycleType.VALIDATION, all_img_count)

  # #Show training images that the YOLO model was trained on...
  if(OPTION_SHOW_TRAINING_IMAGE_GALLERY and MODE == Mode.DEMO):
    if(model_type == ModelType.YOLO_SEG):

      #Get only yolo data and show that...
      yolo_train_ds = fetch_data(model_type, yolo_data, CycleType.TRAINING, all_img_count)
      visualize_training_data(yolo_train_ds, bounding_box_format="xyxy")
    else:
      visualize_training_data(train_ds, bounding_box_format="xyxy")

  #Select the correct backbone for training. Unfortunately, only YOLOv8 modes work at the moment.
  if(MODE == Mode.TRAIN):

    #If this is a YOLO model, then
    if(model_type == ModelType.YOLO):
      yolo_train_model(train_ds, val_ds)     
    else:
      seg_train_model(train_ds, val_ds)

  else:

    #Model type is Yolo-Segmentation
    if(model_type == ModelType.YOLO_SEG):

      #Get index-locked predictions for both yolo and unet data sets
      write_log(-1, "Getting predictions on datasets...")
      do_testing(ModelType.YOLO_SEG, None, CycleType.TESTING, "xyxy", all_img_count, no_metrics=False, yolo_dataset=yolo_data, seg_dataset=seg_data)


    else:
  
      #Run evaluation loop for single model type - mostly for developing both models for YOLO seg
      do_testing(model_type, data, CycleType.TESTING, "xyxy", all_img_count, no_metrics=False)

      #If outputting full test and validation image set for u-net, do that as well
      if(OPTION_GENERATE_FULL_YOLOSEG_OUTPUTMASK_SELECTION):
        if(model_type == ModelType.UNET):

          #This part runs when we're generating masks to save to file. Not used because the results were worse...
          print("Also Generating Training and Validation Masks for UNet... to Send to YOLO")

          #val_data = get_dataset(data, CycleType.VALIDATION, all_img_count)
          #val_img_paths = get_paths_from_dataset(val_data, model_type, ImageDataType.RGB)
          do_testing(model_type, data, CycleType.TESTING, "xyxy", all_img_count, True)

          #train_data = get_dataset(data, CycleType.TRAINING, all_img_count)
          #train_img_paths = get_paths_from_dataset(train_data, model_type, ImageDataType.RGB)
          do_testing(model_type, data, CycleType.TESTING, "xyxy", all_img_count, True)

      

#START
start(MODEL_TYPE)   #Train on YOLO