# Human Pose Classifier — train on your own HD images (MobileNetV2 transfer learning)                                       
import numpy as np                                
import tensorflow as tf                           
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 4: image loader/augmenter
from tensorflow.keras import layers, callbacks     # 5: layers & training callbacks
from tensorflow import keras

train_dir = 'DeepLearning/Human pose images/train'                        # 6: folder with training images (class subfolders)
val_dir = 'DeepLearning/Human pose images/test'                            # 7: folder with validation images (class subfolders)
img_size = (224, 224)                              # 8: model input size (resizes HD images to this)
batch_size = 16                                    # 9: batch size (reduce if OOM)
epochs = 15                                        # 10: number of training epochs
# --------------------------------------------------

# 11: data augmentation for training (rescale + simple augmentations)
train_datagen = ImageDataGenerator( # creates data pipelines
    rescale=1./255,            # 12: scale pixel values 0-255 -> 0.0-1.0
    rotation_range=20,         # 13: small random rotations
    width_shift_range=0.1,     # 14: horizontal shifts
    height_shift_range=0.1,    # 15: vertical shifts
    zoom_range=0.15,           # 16: small zooms
    horizontal_flip=True,      # 17: random flips (if pose classes allow)
    fill_mode='nearest'        # 18: fill missing pixels after transforms
)

# 19: validation generator only rescales (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# 20: create iterator that reads from folders for training
train_gen = train_datagen.flow_from_directory( # reads images from subfolders, auto-assigns labels from folder names.
    train_dir,
    target_size=img_size, # resize each image
    batch_size=batch_size,  
    class_mode='categorical',  # multi-class classification.
    shuffle=True # randomizes order each epoch (good for training)
)

# 22: validation iterator
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 23: number of classes detected from folders
num_classes = train_gen.num_classes

# 24: load MobileNetV2 (pretrained on ImageNet), without top classifier
base_model = tf.keras.applications.MobileNetV2( # is a pretrained CNN (trained on ImageNet). Advantages:
# learns low-level visual features (edges, textures) already — you don’t need to train from scratch.
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False, # removes the original classification head so you can add your own head with num_classes outputs.
    weights='imagenet' # loads pretrained weights.
)

# 25: freeze base model weights (train only the head initially)
base_model.trainable = False

# 26 build classifier head on top of the base model
inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))   # input defines input tensor shape. 
x = base_model(inputs, training=False)                         # Passing inputs through base_model produces feature maps.
x = layers.GlobalAveragePooling2D()(x)                         # reduces each feature map to a single number (averaging spatial dimensions), reducing parameters vs flattening.
x = layers.Dropout(0.4)(x)                                     # randomly turns off 40% of units during training to reduce overfitting.
x = layers.Dense(128, activation='relu')(x)                    # Dense layer 128 with relu learns higher-level combinations of features.
outputs = layers.Dense(num_classes, activation='softmax')(x)   # final softmax layer
model = tf.keras.Model(inputs, outputs)                        # assemble model

# 33: compile model with Adam optimizer + categorical crossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 34: show model summary (layers & params)
model.summary()

# 35: useful callbacks: save best model, early stop, reduce LR on plateau
checkpoint_cb = callbacks.ModelCheckpoint('pose_model_best.h5', save_best_only=True, monitor='val_loss')
# saves best model file by monitoring val_loss. save_best_only=True ensures we only keep best weights.
earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# stops training if val_loss does not improve for patience epochs — prevents overfitting and saves time.
# restore_best_weights=True loads best weights after stopping.
reduceLR_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
# reduces learning rate by factor=0.5 if val_loss stops improving for patience=3 epochs — helps escaping plateaus.

# 38: start training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint_cb, earlystop_cb, reduceLR_cb]
)

# 39: save final model
from tensorflow import keras
model.save("DeepLearning/pose_model_final.keras")
model = keras.models.load_model("DeepLearning/pose_model_final.keras")

#model.save("pose_model_final.keras")

#model = keras.models.load_model("pose_model_final.h5")   # if saved in h5


# Now you can use it for predictions
# ---------------- Example: predict a single image ---------------------
from tensorflow.keras.preprocessing import image   # 40: helper to load single image

# Function to predict a single image
def predict_image(model, img_path, img_size, class_labels):
    img = image.load_img(img_path, target_size=img_size)   # Load image & resize
    img_array = image.img_to_array(img) / 255.0           # Convert to array & normalize
    img_array = np.expand_dims(img_array, axis=0)         # Add batch dimension
    pred = model.predict(img_array)                       # Predict
    pred_class = np.argmax(pred, axis=1)[0]               # Get class index
    confidence = float(pred[0][pred_class])               # Get confidence
    print("Predicted:", class_labels[pred_class], "| Confidence:", round(confidence*100, 2), "%")

# Get class labels from training generator
class_labels = list(train_gen.class_indices.keys())  

# Loop for multiple predictions
while True:
    img_path = input("Enter image path for prediction: ")  # User gives image
    predict_image(model, img_path, (224,224), class_labels)

    # Ask user if they want to continue
    choice = input("Do you want to test more images? (yes/no): ").strip().lower()
    if choice in ["no", "exit", "quit"]:
        print("Exiting model testing... Goodbye! 👋")
        break
