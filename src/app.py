import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIGS ---
IMG_SIZE = 224
MODEL_PATH = "/Users/ivanaguilarjr/Documents/programs/projects/tumor-detection/model_unquant.tflite"
#"/Users/ivanaguilarjr/Documents/programs/projects/tumor-detection/tumor_model_mobilenet.tflite" # Path to your TFLite model
  # Path to the TFLite model

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
last_file_path = None


# --- FUNCTIONS ---

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # read in color (BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert to RGB
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize to (224,224)
    img_norm = img_resized.astype(np.float32) / 255.0    # normalize pixels
    img_norm = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 3) # reshape to batch format
    print("Original shape:", img.shape)
    print("Resized shape:", img_resized.shape)

    return img_norm

def highlight_tumor(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = image.copy()
    mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 150000:
            # x, y, w, h = cv2.boundingRect(cnt)
            # aspect_ratio = float(w) / h
            # if .75 < aspect_ratio < 1.2:  # Filter based on aspect ratio
            #     # draw contour on overlay
            cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Extract only tumor region
    tumor_only = cv2.bitwise_and(image, image, mask=mask)
    background_mask = cv2.bitwise_not(mask)
    tumor_only[background_mask == 255] = [0, 0, 0]  # black background
    tumor_only[mask == 255] = [0, 0, 255]           # red tumor

    return overlay, tumor_only


def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    global last_file_path
    last_file_path = file_path
    print(f"Selected file: {file_path}")

    # Preprocess for model
    processed_img = preprocess_image(file_path)
    input_type = input_details[0]['dtype']
    interpreter.set_tensor(input_details[0]['index'], processed_img.astype(input_type))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    result = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
    result_label.config(text=f"Result: {result}", fg="red" if "Tumor" in result else "green")

    # Show original
    original = Image.open(file_path).resize((250, 250))
    original_tk = ImageTk.PhotoImage(original)
    original_label.config(image=original_tk)
    original_label.image = original_tk

    # Show tumor-highlighted image
    if "Tumor" in result:
        overlay, tumor_only = highlight_tumor(file_path)
        window.tumor_only_image = tumor_only
        window.tumor_overlay_image = overlay
    else:
        overlay = cv2.imread(file_path)
        window.tumor_only_image = overlay
        window.tumor_overlay_image = overlay

    overlay = cv2.resize(overlay, (250, 250))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay)
    overlay_tk = ImageTk.PhotoImage(overlay_pil)
    tumor_label.config(image=overlay_tk)
    tumor_label.image = overlay_tk



def show_only_tumor():
    if not hasattr(window, 'tumor_only_image'):
        return

    tumor = window.tumor_only_image
    tumor = cv2.resize(tumor, (250, 250))
    tumor = cv2.cvtColor(tumor, cv2.COLOR_BGR2RGB)
    tumor_pil = Image.fromarray(tumor)
    tumor_tk = ImageTk.PhotoImage(tumor_pil)

    tumor_window = tk.Toplevel(window)
    tumor_window.title("Tumor Region Only")
    label = Label(tumor_window, image=tumor_tk)
    label.image = tumor_tk
    label.pack(padx=10, pady=10)

def edit_image():
    if not hasattr(window, 'tumor_only_image'):
        return
    
    overlay = window.tumor_overlay_image
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_rgb)
    plt.title("Tumor Highlighted Image")
    plt.axis('on')
    plt.show()

# --- GUI SETUP ---

window = tk.Tk()
window.title("MRI Tumor Detector")
window.geometry("600x600")
window.configure(bg="#323232")
window.resizable(False, False)

title = Label(window, text="MRI Tumor Detection", font=("Arial", 35, "bold"), bg="#323232", fg="#FFFFFF")
title.pack(pady=10)

subtitle = Label(window, text="by Ivan Aguilar", font=("Arial", 18), bg="#323232", fg="#FFFFFF")
subtitle.pack(pady=0)

labelO = Label(window, text="Original Image", font=("Arial", 16), bg="#323232", fg="#FFFFFF")
labelO.place(x=100, y=380)

labelT = Label(window, text="Tumor Highlighted", font=("Arial", 16), bg="#323232", fg="#FFFFFF")
labelT.place(x=380, y=380)

upload_btn = Button(window, text="UPLOAD", command=upload_image, font=("Arial", 20, "bold"), bg="#323232", fg="#000000", activebackground="#323232", bd= 0, highlightthickness=0, width=10, height=2, padx=10, pady=10)
upload_btn.place(x= 30, y= 570, anchor='sw') #bottom left corner.

result_label = Label(window, text="Result: ", font=("Arial", 16, "bold"), bg="#f0f0f0"  , fg="#000000", width=30, height=2)
result_label.place(x=300, y=450, anchor='center')

original_label = Label(window, bg="#323232")
#original_label.pack(side=tk.LEFT, padx=20)
original_label.place(x=30, y=120)

tumor_label = Label(window, bg="#323232")
#tumor_label.pack(side=tk.RIGHT, padx=20)
tumor_label.place(x=320, y=120)

show_tumor_btn = Button(window, text="Show Tumor", command=show_only_tumor, font=("Arial", 20, "bold"), bg="#323232", fg="#000000",activebackground="#323232", bd=0, highlightthickness=0, width=10, height=2, padx=10, pady=10)
show_tumor_btn.place(x=400, y=500)

edit_tumor_btn = Button(window, text="Edit Image", command=edit_image, font=("Arial", 20, "bold"), bg="#323232", fg="#000000", activebackground="#323232", bd=0, highlightthickness=0, width=10, height=2, padx=10, pady=10)
edit_tumor_btn.place(x=215, y=500)

# --- RUN THE APP ---
window.mainloop()