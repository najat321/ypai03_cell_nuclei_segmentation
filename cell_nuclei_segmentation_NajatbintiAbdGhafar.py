#%%
#1. Import packages
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, os

filepath = r"C:\Users\USER\Desktop\deep_learning_computer_vision\semantic_segmentation\cell_neuclei\train"
images = []
masks = []
#%%
#Use os.listdir() method to list down all the image file, then use a for loop to read the images.

#2. Load images
image_path = os.path.join(filepath,'inputs')
for img in os.listdir(image_path):
    #Get the full path of the image file
    full_path = os.path.join(image_path,img)
    #Read the image file based on the full path
    img_np = cv2.imread(full_path)
    #Convert the image from bgr to rgb
    img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    #Resize the image into 128x128
    img_np = cv2.resize(img_np,(128,128))
    #Place the image into the empty list
    images.append(img_np)

#3. Load masks
mask_path = os.path.join(filepath,'masks')
for mask in os.listdir(mask_path):
    #Get the full path of the mask file
    full_path = os.path.join(mask_path,mask)
    #Read the mask file as a grayscale image
    mask_np = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
    #Resize the image into 128x128
    mask_np = cv2.resize(mask_np,(128,128))
    #Place the mask into the empty list
    masks.append(mask_np)

# %%
#4. Convert the list of np array into a full np array
images_np = np.array(images)
masks_np = np.array(masks)
# %%
#5. Data preprocessing
#5.1. Expand the mask dimension to include the channel axis
masks_np_exp = np.expand_dims(masks_np,axis=-1)
#5.2. Convert the mask value into just 0 and 1
converted_masks_np = np.round(masks_np_exp/255)
#5.3. Normalize the images pixel value
normalized_images_np = images_np/255.0
# %%
#6. Perform train test split
from sklearn.model_selection import train_test_split
SEED = 12345
X_train,X_test,y_train,y_test = train_test_split(normalized_images_np,converted_masks_np,shuffle=True,random_state=SEED)

# %%
#7. Convert the numpy array into tensorflow tensors
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)
# %%
#8. Combine features and labels together to form a zip dataset
train = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))
# %%
#9. Convert this into prefetch dataset
#9.1 Define hyperparameters for the tensorflow dataset
TRAIN_LENGTH = len(list(X_train_tensor))
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#%%
#9.2 Create a data augmentation layer through creating a custom class
from tensorflow import keras
class Augment(keras.layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode='horizontal',seed=seed)

    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
# %%
#9.3 Build the dataset
train_batches = (
    train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test.batch(BATCH_SIZE)
# %%
#10. Inspect some data
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ["Input Image","True Mask","Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for images,masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
# %%
#11. Model development
"""
The plan is to apply transfer learning by using a pretrained model as the feature extractor.
Then, proceed to build my own upsampling path with the tensorflow_example module that was just imported + other default keras layers.
"""
#11.1. Use a pretrained model as feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
base_model.summary()
# %%
#11.2. Specify the layers that is needed as outputs for the feature extractor
layer_names = [
    "block_1_expand_relu",      #64x64
    "block_3_expand_relu",      #32x32
    "block_6_expand_relu",      #16x16
    "block_13_expand_relu",     #8x8
    "block_16_project"          #4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#11.3. Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

#11.4. Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),        #4x4  --> 8x8
    pix2pix.upsample(256,3),        #8x8  --> 16x16
    pix2pix.upsample(128,3),        #16x16 --> 32x32
    pix2pix.upsample(64,3)          #32x32 --> 64x64
]

#11.5. Define a function for the unet creation.
def unet(output_channels:int):
    """
    Use functional API to connect the downstack and upstack properly
    """
    #(A) Input layer
    inputs = keras.Input(shape=[128,128,3])
    #(B) Down stack (Feature extractor)
    skips = down_stack(inputs)
    x = skips[-1]       #This is the output that will progress until the end of the model
    skips = reversed(skips[:-1])

    #(C) Build the upsampling path
    """
    1. Let the final output from the down stack flow through the up stack
    2. Concatenate the output properly by following the structure of the U-Net
    """
    for up,skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])

    #(D) Use a transpose convolution layer to perform one last upsampling. This convolution layer will become the output layer as well.
    last = keras.layers.Conv2DTranspose(output_channels,kernel_size=3,strides=2,padding='same')     #64x64 --> 128x128
    outputs = last(x)
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model
# %%
#11.6. Create the U-Net model by using the function
OUTPUT_CLASSES = 3
model = unet(OUTPUT_CLASSES)
model.summary()
keras.utils.plot_model(model)
# %%
#12. Compile the model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
# %%
#13. Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]       #equivalent to tf.expand_dims()
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()
# %%
# #14. Create a custom callback function to display results during model training
# class DisplayCallback(keras.callbacks.Callback):
#     def on_epoch_end(self,epoch,logs=None):
#         #clear_output(wait=True)
#         show_predictions()
#         print('\nSample prediction after epoch #{}\n'.format(epoch+1))
#%%
#14. Create a TensorBoard callback object for the usage of TensorBoard
import datetime
from tensorflow.keras import callbacks
base_log_path = r"tensorboard_logs\cell_nuclei_segementation"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#15. Model training
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(list(X_test_tensor))//BATCH_SIZE//VAL_SUBSPLITS
history = model.fit(train_batches,validation_data=test_batches,validation_steps=VALIDATION_STEPS,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[tb])
# %%
#16. Model deployment
show_predictions(test_batches,3)
# %%
#Model save path
model_save_path = os.path.join(os.getcwd(),"cell_nuclei_segmentation_model.h5")
keras.models.save_model(model,model_save_path)
#%%
#Check if the model can be loaded
model_loaded = keras.models.load_model(model_save_path)
# %%
