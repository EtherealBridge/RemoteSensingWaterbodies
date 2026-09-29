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

#Torch optimizer types
from tensorflow.keras.optimizers import SGD, Adam


#from scipy import ndimage

#For drawing the rectangle
import matplotlib.patches as patches
                                                             #For more image editing functions

#More image manipulation
from skimage.transform import rescale, resize, downscale_local_mean

import plotly.express as px
import plotly.graph_objects as go

from skimage import data, io
from matplotlib import pyplot as plt
import matplotlib.pyplot as pyp






from enum import Enum                           #Enumerations

import os                                     #For image processing and listing out files in the Collab mounted drive

#-----------------------------------#
#Enum
#-----------------------------------#

#Different optimizer options
class Optimizer(Enum):
  ADAM = 0
  RMSPROP = 1

#Image color channels to use for image classification
class ImageColorChannelMode(Enum):
  RGB = 0
  GREYSCALE = 1
  RGBA = 2

#Different water feature types
class Feature(Enum):
  SMALLWATER = 0
  MEDWATER = 1
  LARGEWATER = 0

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
class IOUMetricType(Enum):
  TOTALIOU = 0
  AVGIOU = 1
  CLASSACCURACY = 2
  PIXELACCURACY = 3

#-----------------------------------#
#Properties
#-----------------------------------#


#Application mode
#Mode.TRAIN - For training a new model. Saves to model.keras
#Mode.DEMO - Once the model is trained, for getting image selections and predictions
#MODE = Mode.TRAIN
MODE = Mode.DEMO

#Set the filepath (for saving and loading)
model_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/models/model.keras"
csv_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/logs/output.csv"
bounding_box_annotations = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/annotations/_annotations.csv"
rgb_path = "/home/chrissy/Documents/School/CS547/AIStuff/capstone/data/"

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

#Number of training epochs (loops) to run
EPOCHS = 2

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
CLASS_LIST = ("Lake","Ocean","Pool","River","Wetlands")
class_mapping = dict(zip(range(len(CLASS_LIST)), CLASS_LIST))

#The max size of each class (final class is not featured because anything over dim 1 is the final class)
#CLASS_CUTOFF_DEF_S = 4
#CLASS_CUTOFF_DEF_M = 64
#CLASS_CUTOFF_DEF_L = 300

#Number of classes in this problem. Though we're looking at water vs. not water, we're really just looking for water, so, one class.
# Non-water is just "null" in a logical sense and not a class.
class_count = len(CLASS_LIST)

#---IMAGE PROCESSING STUFF... ---#
#Fetch RGB channels and setr alpha as 1 (as, none of the imported images have an alpha (transparency) channel anyways)
IMAGE_COLOR_CHANNEL_MODE = ImageColorChannelMode.RGB

#The size of edges to cut off each processed image
#This is data augmentation to accomodate for issues with the training data. It had weird black stripes on the edge of it.
IMAGE_CUT_EDGES_SIZE = 4

#The size of what constitutes a "small hole" - anything of this size x,y either dim gets proc'd out.
#SMALL_HOLE_SIZE = 2

#What consitutes a "small object" for bounding box creation? Anything smaller we don't care about.
#SMALL_OBJECT_SIZE = 2

#Data Augmentation Properties
AUGMENTATION_FLIP_VERTICALLY = True
AUGMENTATION_FLIP_HORIZONTALLY = True

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
      self.imgIndex
      self.classification = classification
      self.x0 = x0
      self.y0 = y0
      self.x1 = x1
      self.y1 = y1

  #Get the starting pixel position on the image
  def getStart(self):
    return (x0,y0)

  #Gets the center of this bounding box
  def getCenter(self):
    distx = x1 - x0
    distxc = distx / 2
    disty = y1 - y0
    distyc = disty / 2
    return (x0 + distxc,y0 + distyc)


  #Gets the en (upper right corner) of this bounding box
  def getEnd(self):
    return (x1,y1)

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



#Given the index of a bounding box, g
def getBoundingBoxColor(index):
  return BOUNDING_BOX_DISPLAY_COLORS[index % len(BOUNDING_BOX_DISPLAY_COLORS)]

#Function that resizes image to FILESIZExFILESIZE coordinates with bilinear filtering. pad_to_aspect_ratio required for BB. BB format is xyxy
resizing = keras_cv.layers.Resizing(FILESIZE,FILESIZE,"bilinear",pad_to_aspect_ratio=True,bounding_box_format="xyxy")
  
#Of four values, get the max value
def getMax(v0, v1, v2, v3):
  winner = v0
  if(v1 > winner): winner = v1
  if(v2 > winner): winner = v2
  if(v3 > winner): winner = v3
  return winner

#Of four values, get the minimum value
def getMin(v0, v1, v2, v3):
  winner = v0
  if(v1 < winner): winner = v1
  if(v2 < winner): winner = v2
  if(v3 < winner): winner = v3
  return winner

#Loads an image during training, validation, or testing from the COCO process callback
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=INPUT_CHANNELS)
    return image

#Given the image path, classes, and bounding boxes for a row, loads the image and then combines the image, and bounding boxes
#Class is packed into the bounding box record
def load_dataset(image_path, classes, bbox):
  # Read Image
  image = load_image(image_path)
  bounding_boxes = {
      "classes": tf.cast(classes, dtype=tf.float32),
      "boxes": bbox,
  }
  return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


#Shows dataset records for the given inputs
#To be honest, I couldn't get this to work well and gave up on it, but, I'll try this again later after I hand the assignment in.
def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=1,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


#Converts from dictionary type to tuple type
#https://keras.io/examples/vision/yolov8/
def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=MAX_ANCHOR_BOXES
    )


#Writes the log to file with the correct epoch. Use TESTING if testing.
def write_log(epoch, logstr):
  try:
      with open(csv_path, 'a') as file:
        file.write(logstr + '\n')
  except Exception as e:
      print("Could not write epoch to file : " + str(epoch))

#Run the evaluation for the model
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
        logstr = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + "LOGS for epoch " + str(epoch) + " = " + str(logs).replace(",","")
        #print("OUTPUT=" + logstr)

        """
        try:
          with open(csv_path, 'a') as file:
              file.write(logstr + '\n')
        except Exception as e:
          print("Could not write epoch to file : " + str(epoch))
        """
        
        write_log(epoch, logstr)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs


"""
#Calculate the intersection over union (IOU) for two bounding boxes. 0.0 = no intersection. 1.0 = full intersection (one box is redundant)
def calculate_iou():
  intersection = getIntersectionArea(bounding_box_0, bounding_box_1)
  union = (bounding_box_0.getArea() + bounding_box_1.getArea()) - intersection
  return intersection / union
"""

# #The dice loss of the predicted data vs. the acutal, defined as 2*intersection divided by (sum(y_pred) + sum(y_actual))
# def get_dice_loss(y_pred, y_actual):


# #https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/
# #Use the y predicted and y actual values to get the f1 score, which is defined by 
# # (2 x ((Precision * Recall ) / (Precision + Recall)))
# def get_f1_score(y_pred, y_actual):

# #Calculate the recall based on the predicted values vs. actual values
# def get_recall(y_pred, y_actual):

# #Calculate the precision based on the predicted values vs. actual values
# def get_precision(y_pred, y_actual):


#Gets the area of the square in which two bounding boxes intersect. 0.0 = no intersection at all
def get_iou(start_predx, start_predy, start_actx, start_acty, end_predx, end_predy, end_actx, end_acty):

  #https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

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

  # compute the area of both AABBs
  pred_area = (end_predx - start_predx) * (end_predy - start_predy)
  act_area = (end_actx - start_actx) * (end_acty - start_acty)

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  #Add 0.000001 to denominator to prevent divide by zero error if denominator is 0
  denom = float(pred_area + act_area - intersection_area)
  if(denom == 0.0): denom = denom +  + 0.000001
  iou = intersection_area / denom
  return iou



#Given y_predicted values and y_actual values, calculate the IOU
#y_pred - The predicted values
#y_actual - The actual values
#metric_type - Should the metric here be the average iou? 
#specific_class - If this is for a specific class, the class number. -1 = all classes TODO
def calculate_iou_metric(y_pred, y_actual, metric_type = IOUMetricType.TOTALIOU, specific_class=-1):

  pred_classes = y_pred["classes"]
  pred_boxes = y_pred["boxes"]

  actual_classes = y_actual["classes"]
  actual_boxes = y_actual["boxes"]

  #Total iou, incremented from below
  total_iou = 0.0
  total_box_count = 0

  #For the set of all predictions, index-locked to their image index...
  for i in range(0, len(pred_boxes)):
    pred_box = pred_boxes[i]
    act_box = actual_boxes[i]

    #Add IOU for all existing boxes
    lenP = len(pred_box)
    lenA = len(act_box)
    aLen = min(lenP, lenA)
    
    #Get the max length (highest array) where the values are not -1 (empty)
    maxLen = max(lenP, lenA)

    #Loops through and gets the highest index that is not -1 (empty) in predicted and actual boxes
    countA = 0
    countP = 0
    for a in range(0, maxLen):
      if(a < lenP):
        segP = pred_box[a]
        if(segP[0] != -1.0):
          countP = countP + 1
      if(a < lenA):
        segA = act_box[a]
        if(segA[0] != -1.0):
          countA = countA + 1

    #Get the maximum of the predicted and actual counts - that will be the total box count
    box_count = max(countA,countP)

    #Add to the total number of boxes
    total_box_count += box_count

    #For each box inside a given image / predicted box set
    for a in range(0, box_count):

      #Class restriction

      #Get the current predicted box and the current actual box
      if(a < countP):
        segP = pred_box[a]
      else:
        segP = [-1.0,-1.0,-1.0,-1.0]
      
      if(a < countA):
        segA = act_box[a]
      else:
        segA = [-1.0,-1.0,-1.0,-1.0]

      #Only check IOU if both boxes have real values
      if((len(segP) < aLen) and (len(segA) < aLen)):
        #print("sP = " + str(segP))
        #print("sA = " + str(segA))
        x0P = segP[0]
        y0P = segP[1]
        x1P = segP[2]
        y1P = segP[3]
        x0A = segA[0]
        y0A = segA[1]
        x1A = segA[2]
        y1A = segA[3]
        #If is not an empty array section, then
        if(x0P != -1.0 and x0A != -1):        

          iou = get_iou(x0P, y0P, x0A, y0A, x1P, y1P, x1A, y1A)
          total_iou += iou

  if(metric_type == IOUMetricType.AVGIOU):
    print("Total Box Count = " + str(total_box_count))
    if(total_box_count > 0):
      return total_iou / total_box_count
    else:
      return 0.0
  else:
    return total_iou

#Calculates the total IOU
#y_pred - The predicted values
#y_actual - The actual values
#specific_class - (optional) - The specific class to calculate average iou for
#returns - The total iou
def get_total_iou(y_pred, y_actual, specific_class = -1):
  return calculate_iou_metric(y_pred, y_actual, IOUMetricType.TOTALIOU, specific_class)

#Calculates the average IOU
#y_pred - The predicted values
#y_actual - The actual values
#specific_class - (optional) - The specific class to calculate average iou for
#returns - The average iou
def get_avg_iou(y_pred, y_actual, specific_class = -1):
  return calculate_iou_metric(y_pred, y_actual, IOUMetricType.AVGIOU, specific_class)

#Given the class name, convert it to the class index
#classname (string) - The name of the class to convert to an index - must match values in CLASS_LIST
def getClassIndexFromName(classname) -> int:
  for i in range(0,len(CLASS_LIST)):
    current_class_name = CLASS_LIST[i]
    if(current_class_name == classname):
      return i
  
  #Class name has not been found
  return -1

#-----------------------------------#
# Entry Point - Where the code actually starts doing stuff...
#-----------------------------------#

#Put start message in logs
ts = time.time()
logstr = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + " --- STARTED!"
write_log(-1, logstr)


#Get all files in the repo path
all_rgb_files = os.listdir(rgb_path)
img_paths = []
img_full_paths = []
img_files = []


####
#Image Loading / Processing
####

#Select only image files from all files
for genericFile in os.listdir(rgb_path):
  name, fileExtension = os.path.splitext(genericFile)
  if fileExtension == ".jpg" or fileExtension == ".jpeg" or fileExtension == ".png" or fileExtension == ".bmp":
    img_paths.append(genericFile)
    img_full_paths.append(rgb_path + genericFile)

#Get count of training data
all_img_count = len(img_paths)
print("Total Images = " + str(all_img_count))

#Special note: Training data is [TRAINING_DATA_PCT] of the image set data. Images will be
# SHUFFLED every training data set by using random.shuffle(x)
test_data_count = int(all_img_count * TEST_DATA_PCT)
training_img_count = int(all_img_count * (1.0 - TEST_DATA_PCT))

#Create temp class list
classes = []

#List of image paths
paths = []

# Read the entire content of the bounding box file
lines = read_text_file(bounding_box_annotations)

bounding_boxes = []

#Populate bounding boxes array and class array per image file list item
for img_index in range(0,len(img_full_paths)):
  bounding_boxes.append([])
  #classes.append([0])
  classes.append([])

"""
#TEST - SHOW IMAGE TO SEE BOUNDING BOX DRAWN
fig, ax = plt.subplots()
ax.imshow(mask_img, cmap='gray')
"""
  
#For each bounding box found, add the bounding box for this image
for line in lines:
    
  if(not line.startswith("filename")):
    vals = line.split(",")      
    filename = vals[0]

    #Get the class name
    classname = vals[3]

    #Get bounding box class index from its name
    class_index = getClassIndexFromName(classname)
    
    #Get image index from filename
    #https://www.geeksforgeeks.org/python/how-to-find-the-index-of-value-in-numpy-array/
    #img_index = np.where(img_full_paths == filename)[0]
    img_index = -1
    for i in range(0,len(img_full_paths)):
      img_path = img_full_paths[i]
      if(filename in img_path):
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


print("BBOX = " + str(bounding_boxes[0]))

"""
#For each remaining bounding box, show
for i in range(0, len(boundingBoxes)):
  bb = boundingBoxes[i]


  #Test - show bounding boxes
  #Get the bounding box color to show
  bbcol = getBoundingBoxColor(i)
    
  #Create the rectangle
  rec = plt.Rectangle((bb.x,bb.y), bb.w, bb.h, edgecolor=bbcol, linewidth=2, fill=False)
  ax.add_patch(rec)

plt.show()
"""

#No longer need mask file data and image paths
all_rgb_files = None

#https://keras.io/examples/vision/yolov8/
print("Structuring Data...")
#Create dataframes for x and y data so that they can be processed like normal dataframes
#x_df = pd.DataFrame(x_data)
#y_df = pd.DataFrame(y_data)
image_paths = tf.ragged.constant(img_full_paths)
bbox = tf.ragged.constant(bounding_boxes)
classes = tf.ragged.constant(classes)

#print("IMG PATHS = " + str(image_paths))

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

# Split the dataset into train and validation sets
test_data = data.take(test_data_count)

#Get the rest of the data that is not test data in this instance
rest_of_data = data.skip(test_data_count)

#Get validation size
validation_size = int(len(rest_of_data) * VALIDATION_SPLIT_PCT)
val_data = rest_of_data.take(validation_size)
train_data = rest_of_data.skip(validation_size)

print("Testing Data Size = " + str(len(test_data)))
print("Training Data Size = " + str(len(train_data)))
print("Validation Data Size = " + str(len(val_data)))


####
#Data Augmentation - Random Image Flips, Random Resizing
####
#https://keras.io/examples/vision/yolov8/
layers = []
if(AUGMENTATION_FLIP_VERTICALLY):
  layers.append(keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"))
if(AUGMENTATION_FLIP_HORIZONTALLY):
  layers.append(keras_cv.layers.RandomFlip(mode="vertical", bounding_box_format="xyxy"))

layers.append(keras_cv.layers.Resizing(FILESIZE,FILESIZE,"bilinear",pad_to_aspect_ratio=True,bounding_box_format="xyxy"))
augmenter = keras.Sequential(layers)

####
# Dataset Creation
####

#TODO: JUST HAVE IT shuffle() instead of specifying batch size?
#TODO IS THIS LOADING ALL IMAGES OR JUST A BATCH OF 4 PER CYCLE
#Load training data
#https://keras.io/examples/vision/yolov8/
train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

#Load validation data
val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

#Load testing data
testing_len = len(test_data)
test_ds = test_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.shuffle(testing_len)
test_ds = test_ds.ragged_batch(testing_len, drop_remainder=False)
test_ds = test_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)



#TEST - Shows the training and validation bounding boxes. This works but not well. I messed something up with it.
"""
visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)


visualize_dataset(
    val_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)

"""

"""

visualize_dataset(
    test_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=4, cols=4
)

"""


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


#https://keras.io/examples/vision/yolov8/
train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

#Select the correct backbone for training. Unfortunately, only YOLOv8 modes work at the moment.
if(MODE == Mode.TRAIN):

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

else:
   

  #https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
  loaded_model = keras.saving.load_model(model_path)

  #https://keras.io/examples/vision/yolov8/
  def visualize_detections(model, dataset, bounding_box_format):

      print("Processing Testing Images...")
      images, y_true = next(iter(dataset.take(1)))
      y_pred = model.predict(images)
      #print("Prediction = " + print(str(y_pred)) + ", Actual = " + print(str(y_true)))
      
      #TODO: ONLY FINDS FOUR IMAGES
      #IS POTENTIALLY BECAUSE THE DATASET IS BATCHED - ONLY PULLS 4 IMAGES
      #Get and display the number of images
      num_images = len(images)
      print("There are " + str(num_images) + "images")


      print("Calculating total iou...")

      #Calculate average IOU for predicted vs. actual

      total_iou = get_total_iou(y_pred, y_true, -1)
      total_iou_str = "TESTING - Total IOU = " + str(total_iou)

      #Calculate the total IOU for predicted vs. actual
      avg_iou = get_avg_iou(y_pred, y_true, -1)
      avg_iou_str = "TESTING - Avg IOU = " + str(avg_iou)

      print(total_iou_str)
      print(avg_iou_str)
      write_log("Testing", total_iou_str)
      write_log("Testing", avg_iou_str)

      #https://zihaogeng.medium.com/how-to-evaluate-an-object-detection-model-iou-precision-recall-and-map-f7cc12e0dcf6

      # #Get the precision scores for predicted values vs. actual values
      # precision = get_precision(y_pred, y_true)

      # #Get the recall for predicted values vs. actual values
      # recall = get_recall(y_pred, y_true)

      # #Get the predicted F1 score for predictions vs. actual values
      # f1_score = get_f1_score(y_pred, y_true)





      #TODO: ALLOW FOR PULLING IOU PER CLASS
      print("Calculating iou per class...")

    


      #Note - rows / columns 
      visualization.plot_bounding_box_gallery(
          images,
          value_range=(0, 255),
          bounding_box_format=bounding_box_format,
          y_true=y_true,
          y_pred=y_pred,
          scale=1,
          rows=2,
          cols=2,
          show=True,
          font_scale=1,
          class_mapping=class_mapping,
      )

  #Test data count
  print("Test data count = " + str(len(test_ds)))

  #TODO - LOAD IMAGES IN DIFFERENT WAY? - ALSO CAUSING METRICS TO BE INCORRECT
  #Run evaluation loop
  visualize_detections(loaded_model, dataset=test_ds, bounding_box_format="xyxy")