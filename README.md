# 🕌 Islamic Pose Detector using CNN

This project detects **Islamic prayer poses** (e.g., Sajdah, Dua) using **Deep Learning** with **Convolutional Neural Networks (CNN)**.  
The model is built on **MobileNetV2** (pretrained on ImageNet) and fine-tuned for pose classification.  

It uses **TensorFlow** and **Keras**, along with **ImageDataGenerator** for image preprocessing and augmentation.

---

## 🌟 Features

- Built with **MobileNetV2** (transfer learning)
- Data preprocessing & augmentation using **ImageDataGenerator**
- Training/validation split (80% training, 20% testing)
- Model optimization with:
  - **EarlyStopping**
  - **ModelCheckpoint**
  - **ReduceLROnPlateau**
- Real-time image prediction via user input loop
- Supports any custom Islamic pose dataset (images in folders)
- Automatically detects number of classes from dataset

---

## 🧠 Model Architecture

1. **Base Model**: MobileNetV2 (pretrained, no top classifier)
2. **Custom Head Layers**:
   - Global Average Pooling
   - Dense layer(s) with ReLU activation
   - Output layer with Softmax activation
3. **Optimizer**: Adam  
4. **Loss Function**: Categorical Crossentropy  
5. **Metrics**: Accuracy
---

## ⚙️ Libraries Used

- `numpy`  
- `warnings`  
- `tensorflow`  
- `keras`  
  - `ImageDataGenerator`
  - `layers`
  - `callbacks`

---

## 🚀 How It Works

1. **Ignore Warnings** — for cleaner output.  
2. **Load Dataset** — using `ImageDataGenerator` for training and validation.  
3. **Apply Augmentation** — rotation, zoom, shift, etc. on training data.  
4. **Load MobileNetV2** — without top layers, frozen base weights.  
5. **Add Custom Layers** — for pose classification.  
6. **Train Model** — with callbacks for best performance.  
7. **Evaluate & Save Model** — in `.keras` format.  
8. **Predict New Images** — using custom prediction function.

---

## 🧩 Prediction Function

This function allows the user to input an image path, and the model predicts which Islamic pose (e.g., Dua, Sajdah) the image represents.

After showing the prediction and confidence score, the program asks the user if they want to test more images.
If the user enters “yes”, the process repeats.
If the user enters “no”, “exit”, or “quit”, the program ends with a goodbye message.
