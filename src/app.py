import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from run import tiff_call
from skimage import io as io_
from tensorflow.keras import backend as K

from predict import dice_coef, iou, dice_coef_loss


#=============================== App Header ===================================#
head, photo = st.columns(2)    
with head:   
    st.title("HeadHunter - Brain Tumor Detector")



#============================= Model Loading ===================================#
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("./models/unet_brain_mri_seg.hdf5", 
                                        custom_objects={'dice_coef_loss': dice_coef_loss, 
                                                        'iou': iou, 
                                                        'dice_coef': dice_coef})
    return model

with st.spinner('Loading model into memory...'):
    model = load_model()


#============================== FILE UPLOADER ===================================#
def load_and_prep_image(image):
    """
    Reads an image from filename, and preprocessess 
    it according to the model.
    """
    img = io_.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def file_Uploader():
    file = st.file_uploader("Upload file", type=["png", "jpeg", "jpg"])
    show_file = st.empty()

    if not file:
        show_file.info("Upload image of Brain MRI.")
        return

    content = file.name
    
    path = tiff_call(content)

    st.write("Detection is shown in red")
    with st.spinner("Classifying....."):
            img = load_and_prep_image(path)
            pred_mask = model.predict(tf.expand_dims(img, axis=0)/255)
            pred_mask = pred_mask[np.newaxis, :, :, :]
            pred_mask = np.squeeze(pred_mask) > .5

            img[pred_mask == 1] = (255, 0, 0)

    st.write("")
    st.image(img, caption="Detected Tumor", use_column_width=True)

file_Uploader()
