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

#Import numpy for math functionsE
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

#Different optimizer options
class Optimizer(Enum):
  ADAM = 0
  RMSPROP = 1

#Image color channels to use for image classification
class ImageColorChannelMode(Enum):
  RGB = 0
  GREYSCALE = 1
  RGBA = 2

#Different optimizer options
class Optimizer(Enum):
  ADAM = 0
  RMSPROP = 1

#Should this code run training or the demo?
class Mode(Enum):
  TRAIN = 0,
  LOADANDTRAIN = 1,
  DEMO = 2,

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

#-----------------------------------#
#Properties
#-----------------------------------#


#Application mode
#Mode.TRAIN - For training a new model. Saves to the model name set
#Mode.DEMO - Once the model is trained, for getting image selections and predictions
MODE = Mode.TRAIN
#MODE = Mode.DEMO

#Select the specific model type - YOLO (bounding box segmentation) or U-NET Xception-Like Model
MODEL_TYPE = ModelType.YOLO
#MODEL_TYPE = ModelType.UNET
#MODEL_TYPE = ModelType.YOLO_SEG

#If true, saves segmentation image predictions to file. If false, skips this step, assuming this had already been done previously, as a shortcut.
#Saving images takes time, but, the output of seg images isn't necessarily deterministic (that I know of), so, this would be True for anything but code debugging (where we want to save time)
OPTION_SAVE_SEG_IMAGES = True

#If true, will calculate metrics and save them to file. If false, will skip this step all together
OPTION_CALCULATE_METRICS = False

#If true, will show the correct output images. If false, will not. Skipping showing the output images will just cause the skip to run and end without a break from the image viewer.
OPTION_SHOW_OUTPUT_IMAGES = True


#Set the filepath (for saving and loading)
yolo_model_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/models/yolo.keras"
unet_model_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/models/unet.keras"
if(MODEL_TYPE == ModelType.YOLO or MODEL_TYPE == ModelType.YOLO_SEG):
  model_path = yolo_model_path
else:
  model_path = unet_model_path
csv_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/logs/output.csv"
metric_base_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/logs/"
bounding_box_annotations_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/annotations/_annotations.csv"

#The output mask and output annotation file for using a combination of segmentation and YOLO
output_mask_folder_name = "output_masks"
output_masks = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/" + output_mask_folder_name + "/"

output_mask_annotation_file = output_masks + "_annotations.csv"

#Get correct image paths...
yolo_img_path ="/home/chrissy/Documents/School/CS547/AIStuff/capstone/seg_images/"
seg_img_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/seg_images/"
yolo_seg_img_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/output_masks/"
seg_checkpoint_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/models/seg_chk.keras"

#YOLO 
if(MODEL_TYPE == ModelType.YOLO):
    image_data_path = yolo_img_path

#UNET
elif(MODEL_TYPE == ModelType.UNET):
    image_data_path = seg_img_path
    segmentation_checkpoint = seg_checkpoint_path

#YOLO_SEG - This location is for testing only / is for the output masks from the segmentation model, saved to file. Those will be used to determine the class of a water body.
else:

    image_data_path = yolo_seg_img_path

#This sets the size based on the YOLO version used
#I would like to make this functional in the future
#Unfortunately, only YOLOv8 versions work at the moment
#The file width and height to convert images to in processing
FILESIZE = 512

#The learning rate for all models
LEARNING_RATE=0.001

#Believe this allows for performing normalization during training? Some Keras pages recommended using it and the tutorial I followed uses it.
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
# er of training epochs (loops) to run
EPOCHS = 500

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

if(MODEL_TYPE == ModelType.YOLO):
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

    #if(MODEL_TYPE == ModelType.YOLO):
        #image = tf.image.decode_png(image, channels=channels)
    #else:
    #image = tf.image.decode_jpeg(image, channels=channels)

    # if(MODEL_TYPE == ModelType.YOLO):
    #   image = tf.image.decode_image(image, channels=channels)

    #   #https://stackoverflow.com/questions/37772329/tensorflow-tensor-set-shape-valueerror-image-must-be-fully-defined
    #   image.set_shape([None,None,1])
    # else: 
    image = tf.image.decode_image(image, channels=channels)
    image.set_shape([None, None, channels])

    #if(MODEL_TYPE == ModelType.UNET):
      #Resize the image to the correct file size
      #image = tf.image.resize(image, (FILESIZE, FILESIZE), method="nearest")

    return image

#Given the image path and mask path for a row, loads the image with its mask image. Runs during any model processing.
def load_seg_data(image_path, mask_path):

  #Load the base rgb image being trained on (3 channels, rgb)
  image = load_image(image_path, 3)

  #Load the mask for the base image (1 channel, grey)
  mask = load_image(mask_path, 1)

  #TODO: DOES THIS PROPERLY RESIZE SEG IMG?
  #Resizing for all images
  img_resizer = keras.layers.Resizing(height=FILESIZE, width=FILESIZE, interpolation="bilinear")
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

#Get the metric file path from the metric type and cycle type
#model_type - The type of model being run
#metric_type - Type of metric used (type of MetricType)
#cycle_type - What cycle is this in? Training? Validation? Testing?
#iou (Optional) - If this metric file is for a specific IoU threshold. If set to -1, is ignored
def get_metric_file_path(model_type, metric_type, cycle_type):
  
  model = "yolo_"
  if(model_type == ModelType.UNET):
      model = "unet_"
    
  prefix = "training_"
  if(cycle_type == CycleType.VALIDATION):
    prefix="validation_"
  if(cycle_type == CycleType.TESTING):
    prefix="testing_"

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

  return metric_base_path + model + prefix + descriptor + ".csv"

#Writes a metric to file
#epoch - The epoch for this metric if applicable
#model_type - The type of model being run
#metric_type - The type of metric this metric entry is for
#cycle_type - Where in the cycle this is? Training? Validation? Testing? See CycleType enum.
#metric_val - The value to output
#iou - If applicable, the iou threshold for this metric. Different iou metrics save to different metric output files.
def write_metric(epoch, model_type, metric_type, cycle_type, metric_val, iou=-1.0):

  #Get the filename to write to
  metric_file_path = get_metric_file_path(metric_type, cycle_type)

  #Write to file
  write_file(epoch, str(metric_type) + "," + str(cycle_type) + "," + str(iou) + "," + str(metric_val), metric_file_path)


#Writes the log to file with the correct epoch.
#epoch - The current epoch if applicable. Otherwise set as -1
#logstr - The log string to write to file
def write_log(epoch, logstr):
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


#Gets the segmentation metrics
#y_pred - The predicted values from the model
#y_actual - The actual values from the model
#specific_class - If looking for a specific class
#metric_type - The type of metric to get
def get_seg_metrics_class(y_pred, y_actual, specific_class, metric_type):

  #Get the maximum prediction index (e.g. 0 if index 0 has the highest prediction of the <class count> predictions
  # Gets the predictions with the index of the BEST prediction in the final column (axis=-1 gets last column)
  y_max = np.argmax(y_pred, -1, keepdims=True)

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
          elif(index_actual == specific_class and index_actual != specific_class):
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


 
#Get the data augumentation layers to use for models and return them as a set of augmentation steps
#file_size - The file size to resize images to
#class_count - Number of classes to use in this model
def get_augmentation_layers(file_size, class_count):
  
  ####
  #Data Augmentation - Random Image Flips, Random Resizing
  ####
  #https://keras.io/examples/vision/yolov8/
  layers = []
  if(MODEL_TYPE == ModelType.YOLO):
    #layers.append(keras_cv.layers.Resizing(file_size,file_size,"bilinear",pad_to_aspect_ratio=True,bounding_box_format="xyxy"))
    if(AUGMENTATION_FLIP_VERTICALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"))
    if(AUGMENTATION_FLIP_HORIZONTALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="vertical", bounding_box_format="xyxy"))
    if(AUGMENTATION_RANDOM_ROTATION):
      layers.append(keras_cv.layers.RandomRotation(factor=(0.0,1.0),bounding_box_format="xyxy", segmentation_classes=class_count))

    layers.append(keras_cv.layers.Resizing(FILESIZE,FILESIZE,"bilinear",pad_to_aspect_ratio=True,bounding_box_format="xyxy"))

  else:
    #layers.append(keras_cv.layers.Resizing(file_size,file_size,"bilinear",pad_to_aspect_ratio=True))
    if(AUGMENTATION_FLIP_VERTICALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="horizontal"))
    if(AUGMENTATION_FLIP_HORIZONTALLY):
      layers.append(keras_cv.layers.RandomFlip(mode="vertical"))
    if(AUGMENTATION_RANDOM_ROTATION):
      layers.append(keras_cv.layers.RandomRotation(factor=(0.0,1.0), segmentation_classes=class_count))

    #TODO: MAKE SURE THIS WORKS WITH SEMANTIC SEG
    layers.append(keras_cv.layers.Resizing(FILESIZE,FILESIZE,"nearest",pad_to_aspect_ratio=True))

  if(AUGMENTATION_RANDOM_GAUSSIAN_BLUR):
    layers.append(keras_cv.layers.RandomGaussianBlur(kernel_size=3,factor=(0.0,0.1)))
  if(AUGMENTATION_RANDOM_COLORJITTER):
    layers.append(keras_cv.layers.RandomColorJitter(value_range=(0,255),brightness_factor=(-0.05,0.05),contrast_factor=(0.0,0.05),saturation_factor=(0.35,0.45), hue_factor=(0.0,0.05)))


  augmenter = keras.Sequential(layers)
  return augmenter

#logs the actual precision and recall per class
#y_pred - The y predicted values (from the model)
#y_actual - The actual ground truth values
def output_metrics_per_class(y_pred,y_actual):

  #Messages to write to the log / print out to the console
  log_messages = []

  #Get overall class metrics

  if(MODEL_TYPE == ModelType.YOLO):

    #IoU Amouts to use as thresholds
    iou_thresholds = [0.5, 0.95]

    #For each IoU threshold, create a distinct report
    for iou_threshold in iou_thresholds:
        
      class_overall_accuracy = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.CLASSACCURACY, iou_threshold)
      box_overall_accuracy = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.BOXACCURACY, iou_threshold)
      log_messages.append("Overall Class Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(class_overall_accuracy))
      log_messages.append("Overall Box Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(box_overall_accuracy))
      write_metric(-1, MetricType.CLASSACCURACY, CycleType.TESTING, class_overall_accuracy, iou_threshold)
      write_metric(-1, MetricType.BOXACCURACY, CycleType.TESTING, box_overall_accuracy, iou_threshold)

      class_overall_precision = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.CLASSPRECISION, iou_threshold)
      box_overall_precision = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.BOXPRECISION, iou_threshold)
      log_messages.append("Overall Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(class_overall_precision))
      log_messages.append("Overall Box Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(box_overall_precision))
      write_metric(-1, MetricType.CLASSPRECISION, CycleType.TESTING, class_overall_precision, iou_threshold)
      write_metric(-1, MetricType.BOXPRECISION, CycleType.TESTING, box_overall_precision, iou_threshold)

      class_overall_recall = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.CLASSRECALL, iou_threshold)
      box_overall_recall = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.BOXRECALL, iou_threshold)
      log_messages.append("Overall lass Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(class_overall_recall))
      log_messages.append("Overall Box Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(box_overall_recall))
      write_metric(-1, MetricType.CLASSRECALL, CycleType.TESTING, class_overall_recall, iou_threshold)
      write_metric(-1, MetricType.BOXRECALL, CycleType.TESTING, box_overall_recall, iou_threshold)

      class_overall_f1 = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.CLASSF1, iou_threshold)
      box_overall_f1 = get_yolo_metrics_class_box(y_pred, y_actual, -1, MetricType.BOXDICE, iou_threshold)
      log_messages.append("Overall Class Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(class_overall_f1))
      log_messages.append("Overall Box Accuracy [IoU=" + str(iou_threshold) + ")] = " + str(box_overall_f1))
      write_metric(-1, MetricType.CLASSF1, CycleType.TESTING, class_overall_f1, iou_threshold)
      write_metric(-1, MetricType.BOXDICE, CycleType.TESTING, box_overall_f1, iou_threshold)

      #Dump all metrics messages to log
      for log_message in log_messages:
        write_log(-1, log_message)
          
      #Reset log register
      log_messages = []

    for class_index in range(0, class_count):
      class_name = CLASS_LIST[class_index]

      class_iou = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.AVGIOU, 0.5)
      write_metric(class_index, MetricType.AVGIOU, CycleType.TESTING, class_iou, 0.0)
      log_messages.append("Class [" + class_name + "] IoU=" + str(class_iou))

      #For each IoU threshold, create a distinct report
      for iou_threshold in iou_thresholds:
        
        #Accuracy metrics
        class_accuracy = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.CLASSACCURACY, iou_threshold)
        box_accuracy = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.BOXACCURACY, iou_threshold)
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Class Accuracy = " + str(class_accuracy))
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Box Accuracy = " + str(box_accuracy))
        write_metric(class_index, MetricType.CLASSACCURACY, CycleType.TESTING, class_accuracy, iou_threshold)
        write_metric(class_index, MetricType.BOXACCURACY, CycleType.TESTING, box_accuracy, iou_threshold)

        #Precision metrics
        class_precision = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.CLASSPRECISION, iou_threshold)
        box_precision = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.BOXPRECISION, iou_threshold)
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Class Precision = " + str(class_precision))
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Box Precision = " + str(box_precision))
        write_metric(class_index, MetricType.CLASSPRECISION, CycleType.TESTING, class_precision, iou_threshold)
        write_metric(class_index, MetricType.BOXPRECISION, CycleType.TESTING, box_precision, iou_threshold)

        #Recall metrics
        class_recall = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.CLASSRECALL, iou_threshold)
        box_recall = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.BOXRECALL, iou_threshold)
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Class Recall = " + str(class_recall))
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Box Recall = " + str(box_recall))
        write_metric(class_index, MetricType.CLASSRECALL, CycleType.TESTING, class_recall, iou_threshold)
        write_metric(class_index, MetricType.BOXRECALL, CycleType.TESTING, box_recall, iou_threshold)

        #F1 Metrics
        class_f1 = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.CLASSF1, iou_threshold)
        box_f1 = get_yolo_metrics_class_box(y_pred, y_actual, class_index, MetricType.BOXDICE, iou_threshold)
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Class Recall = " + str(class_f1))
        log_messages.append("IoU=[" + str(iou_threshold) + "] Class [" + class_name + "] Box Dice = " + str(box_f1))
        write_metric(class_index, MetricType.CLASSF1, CycleType.TESTING, class_f1, iou_threshold)
        write_metric(class_index, MetricType.BOXDICE, CycleType.TESTING, box_f1, iou_threshold)

        #Dump all metrics messages to log
        for log_message in log_messages:
          write_log(-1, log_message)

      #Clear log register:
      log_messages = []

  #Output metrics for segmentation
  else:

    #Get metrics for everything
    write_log(-1, "Metrics: Predicted vs. empty")

    #For each class
    for class_index in range(1, class_count):
      class_name = CLASS_LIST[class_index]
      accuracy = get_seg_metrics_class(y_pred, y_actual, class_index, MetricType.PIXELACCURACY)
      log_messages.append("Class [" + class_name + "] Class Accuracy = " + str(accuracy))
      recall = get_seg_metrics_class(y_pred, y_actual, class_index, MetricType.PIXELRECALL)
      log_messages.append("Class [" + class_name + "] Class Recall = " + str(recall))
      precision = get_seg_metrics_class(y_pred, y_actual, class_index, MetricType.PIXELPRECISION)
      log_messages.append("Class [" + class_name + "] Class Precision = " + str(precision))
      f1_score = get_seg_metrics_class(y_pred, y_actual, class_index, MetricType.PIXELDICE)
      log_messages.append("Class [" + class_name + "] Class Dice = " + str(f1_score))
      
    #Clear log register:
    log_messages = []

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
  callbacks = [keras.callbacks.ModelCheckpoint(filepath=segmentation_checkpoint, save_best_only=True), EvaluateSegMetricsCallback(data=val_ds, save_path=model_path)]

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
  #Set backbone
  #https://keras.io/examples/vision/yolov8/
  backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_m_backbone_coco"  # We will use yolov8 medium backbone with coco weights
  )

    #https://keras.io/examples/vision/yolov8/
  yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
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

  #TODO: WITH MULTIPLE CLASSES, CHANGE TO CATEGORICAL CROSS-ENTROPY
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
  yolo.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS,
      callbacks=[EvaluateCOCOMetricsCallback(val_ds, model_path), keras_cv.callbacks.PyCOCOCallback(validation_data=val_ds, bounding_box_format="xyxy")],
      verbose=VERBOSE,
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
        
        #If the current index doesn't match the previous index, save the image to file
        if(prev_img_index != imgindex):

            #Saves the current segmentation image based on accumulated pixels
            saved_seg_img_path = save_seg_img(image_paths[imgindex], img_pixels)

            #Add the saved segmentation image path to the list of saved segmentation images
            seg_img_paths.append(saved_seg_img_path)

            #Purge currently saved pixels to start writing for new image...
            img_pixels = []

        #Get the current predicted pixel (index 3) from the predictions - the predicted pixel is the calculated max index from the predictions
        predicted_pixel = y_pred[imgindex, x, y]
        
        #Get the rgb value to save for the mask
        rgb_value = (predicted_pixel[0],predicted_pixel[0],predicted_pixel[0])

        #Add pixel to image pixels
        img_pixels.append(rgb_value)

        #Sets the previous image index to this index for next loop
        prev_img_index = imgindex


#TODO: FOR YOLO_SEG, CAN WE JUST USE THE EXISTING ANNOTATIONS FILE?
#https://keras.io/examples/vision/yolov8/
def visualize_detections(img_paths, model, dataset, bounding_box_format):

  #Load the images and either classes / bounding boxes, or prediction images
  print("Processing Testing Images...")
  images, y_actual = next(iter(dataset.take(1)))
  #Get predicted
  y_pred = model.predict(images, batch_size=BATCH_SIZE)

  #If this is a segmentation test save the segmentation images so that we can run YOLO on it next for the classification...
  if(MODEL_TYPE == ModelType.UNET):

      #If saving segmentation images is enabled, then
      if(OPTION_SAVE_SEG_IMAGES):

          #Save the segmentation images
          save_seg_masks(img_paths, y_pred)

  #Output count of boxes
  print("Prediction Count = " + str(len(y_pred)))
  print("Actual Annotation Count = " + str(len(y_actual)))

  #Get and display the number of images
  num_images = len(images)
  print("There are " + str(num_images) + "images")

  if(MODEL_TYPE == ModelType.YOLO):
    print("Calculating total iou...")

    #Calculate the total IOU for predicted vs. actual
    #iou = calculate_iou(y_pred, y_actual)
    #avg_iou_str = "TESTING - IOU = " + str(iou)
    #write_log("Testing", avg_iou_str)

  #If true, will calculate and show the metrics. If false, will do neither and skip this step all together
  if(OPTION_CALCULATE_METRICS):

      #https://zihaogeng.medium.com/how-to-evaluate-an-object-detection-model-iou-precision-recall-and-map-f7cc12e0dcf6
      #Outputs the metrics per class:
      output_metrics_per_class(y_pred, y_actual)

  #If we are showing the output images then...
  if(OPTION_SHOW_OUTPUT_IMAGES):

      #If this model type is yolo, then, show yolo images
      if(MODEL_TYPE == ModelType.YOLO):

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
            show=True,
            font_scale=1,
            class_mapping=class_mapping,
        )

      #If this is image segmentation, then...
      else:

        #Get the predictions with the index of the BEST prediction in the final column (axis=-1 gets last column)
        y_pred_max = np.argmax(y_pred, axis=-1, keepdims=True)

        print("y_pred_max_shape = "  + str(y_pred_max.shape[0]) + "," + str(y_pred_max.shape[1]) + "," + str(y_pred_max.shape[2]) + "," + str(y_pred_max.shape[3]))

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
    
#Prepares the YOLO dataset that will be split into training, validation, test  data
def prepare_yolo_data(mask_full_paths):

  #PROBLEM - IMAGE SEG FILE NAMES AND MASK FILE NAMES DO NOT MATCH THE ANNOTATION FILE NAMES!
  #THEREFORE, THESE CANNOT BE REFERENCED FROM ANNOTATIONS
  #ANNOTATIONS FILE NEEDS TO BE READ DIFFERENTLY - MAYBE JUST THE FIRST PART OF IT? NEED TO IGNORE THE SECOND PART
  # WHERE THERE IS A GUID. THE GUID OF SEG IMAGES DOESNT MATCH THE GUID OF THE YOLO IMAGES

  #LOGIC - 
  #1 - SPLIT THE ANNOTATION FILENAME VALUE BY .
  #2 - USE THE FIRST ELEM FROM THIS ANNOTATION VALUE AND REMOVE _JPG AND _PNG TO SEARCH AGAINST img_full_paths - GET THE FIRST PART OF img_full_paths TOO WITHOUT _JPG OR _PNG
  #3 - WHEN FOUND, USE THOSE BOUNDING BOXES 

  # Read the entire content of the bounding box file
  lines = read_text_file(bounding_box_annotations_path)

  #Create temp class list
  classes = []

  bounding_boxes = []

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
      actual_file_descriptor = file_parts[0]

      #Get the class name
      classname = vals[3]

      #Get bounding box class index from its name
      class_index = getClassIndexFromName(classname)
      
      #Get image index from filename
      #https://www.geeksforgeeks.org/python/how-to-find-the-index-of-value-in-numpy-array/
      #img_index = np.where(img_full_paths == filename)[0]
      #This image is used to add bounding boxes to this image, not to get the path
      img_index = -1
      for i in range(0,len(mask_full_paths)):
        img_path = mask_full_paths[i]

        #Does the file descriptor exist in this image path? If so, this is the correct mask image to use for this YOLO entry...
        if(actual_file_descriptor in img_path):
          img_index = i
          break

      #Verify this is an entry for a valid file among the image files. otherwise, will not be added
      if(img_index > -1):
        x0 =  float(vals[4])
        y0 =  float(vals[5])

        x1 =  float(vals[6])
        y1 =  float(vals[7])

        #Create formatted bounding box
        bbox = BoundingBox(img_index, class_index, x0, y0, x1, y1)

        #Add Bounding Box
        bounding_boxes[img_index].append(bbox.toTensor())
        classes[img_index].append(class_index)
        #print("Appended class " + str(class_index))
        #classes[img_index].append(1)

      else:
        write_log("ERROR", "Mask Img Not Found - Desc=" + actual_file_descriptor)

  #https://keras.io/examples/vision/yolov8/
  write_log(-1, "Structuring Data...")
  image_paths = tf.ragged.constant(mask_full_paths)
  bbox = tf.ragged.constant(bounding_boxes)
  classes = tf.ragged.constant(classes)

  #Create dictionary from tensor "slices" - e.g. image paths, classes bounding boxes
  data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))



  print("Images:", len(mask_full_paths),
      "with boxes:", sum(1 for b in bounding_boxes if len(b) > 0))
      
  #returns the data dictionary that gets split into training, test, etc. data
  return data

#Prepares data for semantic segmentation, which will be split into training, validation, test datasets
def prepare_seg_data(img_full_paths, mask_full_paths):

  image_paths = tf.ragged.constant(img_full_paths)
  mask_paths = tf.ragged.constant(mask_full_paths)
  data = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
  return data


#Print the starting message
def start_message():
  #Put start message in logs
  ts = time.time()
  logstr = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + " --- STARTED!"
  write_log(-1, logstr)

#Check if the settings are being run in a valid state - e.g., can't running training on YOLO_SEG because that's only for testing.
#returns - True if state is valid, or, false if state is invalid
def check_for_valid_state():

    #Invalid state - running training when YOLO_SEG is selected. To train for YOLO, just select ModelType.YOLO instead...
    if(MODE == Mode.TRAIN and MODEL_TYPE == ModelType.YOLO_SEG):
        return False

    #Did not hit any invalid states
    return True

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

  img_full_paths = []
  mask_full_paths = []

  ####
  #Image Loading / Processing
  ####

  #Select mask files from all files
  for genericFile in sorted(os.listdir(image_data_path)):
    name, fileExtension = os.path.splitext(genericFile)
    if fileExtension == ".png":
      mask_full_paths.append(image_data_path + genericFile)

  #Only load mask paths if using semantic segmentation. YOLO uses bounding boxes instead.
  #if(model_type == ModelType.UNET):

  #Select image files from all files
  for genericFile in sorted(os.listdir(image_data_path)):
    name, fileExtension = os.path.splitext(genericFile)
    if fileExtension == ".jpg" or fileExtension == ".jpeg":
      img_full_paths.append(image_data_path + genericFile)

  #TODO: YOLO_SEG IS DEMO ONLY - WE WOULD ONLY WANT TO GET TESTING DATA AND WE DONT NEED TO SPLIT IT! 
  #ONLY DO THAT AND FORGO THE REST OF THE CODE BELOW (MINUS LOADING IMAGES)
  #ITS ERRORING BC ITS TRYING TO LOAD TRAINING AND VALIDATION DATA AS WELL AND GETTING TRAINING / VALIDATION IMAGES
  #FOR YOLO SEG, ONLY LOAD VALIDATION IMAGES...

  #Get count of training data
  if(MODEL_TYPE == ModelType.YOLO or MODEL_TYPE == ModelType.YOLO_SEG):
    all_img_count = len(mask_full_paths)
  else:
    all_img_count = len(img_full_paths)
  write_log(-1, "Total Images = " + str(all_img_count))

  #Show mask count if loading masks from file
  if(model_type == ModelType.UNET):
    all_mask_count = len(mask_full_paths)
    write_log(-1, "Total Masks = " + str(mask_full_paths))

  #Special note: Training data is [TRAINING_DATA_PCT] of the image set data. Images will be
  # SHUFFLED every training data set by using random.shuffle(x)
  test_data_count = int(all_img_count * TEST_DATA_PCT)
  training_img_count = int(all_img_count * (1.0 - TEST_DATA_PCT))

  #Get dataset to use based on the current model type YOLO, U-NET, etc.
  if(model_type == ModelType.YOLO or model_type == ModelType.YOLO_SEG):
    data = prepare_yolo_data(mask_full_paths)
  else:
    data = prepare_seg_data(img_full_paths, mask_full_paths)

 

  # Split the dataset into train and validation sets
  test_data = data.take(test_data_count)

  test_img_paths = []
  test_mask_paths = []
  if(MODEL_TYPE == ModelType.UNET):
    for imgpath, maskpath in test_data:
      test_img_paths.append(imgpath.numpy().decode("utf-8"))
      test_mask_paths.append(maskpath.numpy().decode("utf-8"))
  else:
    for imgpath, boxes, classes in test_data:
      test_img_paths.append(imgpath.numpy().decode("utf-8"))

  #Get the rest of the data that is not test data in this instance
  rest_of_data = data.skip(test_data_count)

  #Get validation size
  validation_size = int(len(rest_of_data) * VALIDATION_SPLIT_PCT)

  #If training, then
  if(MODE == Mode.TRAIN):
    val_data = rest_of_data.take(validation_size)
    train_data = rest_of_data.skip(validation_size)
    print("Training Data Size = " + str(len(train_data)))
    print("Validation Data Size = " + str(len(val_data)))

  print("Testing Data Size = " + str(len(test_data)))


  #Get augmentation layers for Keras models
  augmenter = get_augmentation_layers(FILESIZE, class_count)

  ####
  # Dataset Creation
  ####
  #Load training data
  #load_dataset - loads either the YOLO or UNet dataset based on the current model selected
  #https://keras.io/examples/vision/yolov8/
  if(model_type == ModelType.YOLO or model_type == ModelType.YOLO_SEG):

      #If training, then
    if(MODE == Mode.TRAIN):
      train_ds = train_data.map(load_yolo_data, num_parallel_calls=tf.data.AUTOTUNE)
      val_ds = val_data.map(load_yolo_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
      test_ds = test_data.map(load_yolo_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    #Function that resizes image to FILESIZExFILESIZE coordinates with bilinear filtering. pad_to_aspect_ratio required for BB. BB format is xyxy
    resizing = keras_cv.layers.Resizing(FILESIZE,FILESIZE,"nearest",pad_to_aspect_ratio=True,bounding_box_format="xyxy")
    
  else:
    if(MODE == Mode.TRAIN):
      train_ds = train_data.map(load_seg_data, num_parallel_calls=tf.data.AUTOTUNE)
      val_ds = val_data.map(load_seg_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
      test_ds = test_data.map(load_seg_data, num_parallel_calls=tf.data.AUTOTUNE)

    resizing = keras_cv.layers.Resizing(FILESIZE,FILESIZE,"nearest",pad_to_aspect_ratio=True)

  if(MODE == Mode.TRAIN):
    training_len = len(train_data)
    train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    #Load validation data
    val_len = len(val_data)
    val_ds = val_ds.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)

    val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
  else:
    #Load testing data
    testing_len = len(test_data)
    test_ds = test_ds.shuffle(testing_len)
    test_ds = test_ds.ragged_batch(testing_len, drop_remainder=False)

    test_ds = test_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)



  #Map conversion from dictionary to image/bounding bix tuple for YOLO
  if(model_type == ModelType.YOLO):
    if(MODE == Mode.TRAIN):
      #https://keras.io/examples/vision/yolov8/
      train_ds = train_ds.map(yolo_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
      val_ds = val_ds.map(yolo_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    else:
      test_ds = test_ds.map(yolo_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
  else:
    if(MODE == Mode.TRAIN):
      train_ds = train_ds.map(seg_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
      val_ds = val_ds.map(seg_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    else:
      test_ds = test_ds.map(seg_dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

  #Perform data prefetch

  if(MODE == Mode.TRAIN):
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
  else:
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

  #Select the correct backbone for training. Unfortunately, only YOLOv8 modes work at the moment.
  if(MODE == Mode.TRAIN):

    #If this is a YOLO model, then
    if(model_type == ModelType.YOLO):
      yolo_train_model(train_ds, val_ds)     
    else:
      seg_train_model(train_ds, val_ds)

  else:
    
    #https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
    loaded_model = keras.saving.load_model(model_path)    

    #Test data count
    print("Test data count = " + str(len(test_ds)))

    #TODO - LOAD IMAGES IN DIFFERENT WAY? - ALSO CAUSING METRICS TO BE INCORRECT
    #Run evaluation loop
    visualize_detections(test_img_paths, loaded_model, dataset=test_ds, bounding_box_format="xyxy")

#START
start(MODEL_TYPE)   #Train on YOLO