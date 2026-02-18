import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import glob
import io
import os 
import tempfile
from scipy.ndimage import zoom  
# from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow.keras.utils import to_categorical
from keras.models import load_model 


my_model = load_model('brats_2020_2021_3d_final_250.hdf5', compile=False) 

st.title("🧠 Brain Tumor Segmentation")

# Upload 3 images
flair_file = st.file_uploader("Upload FLAIR MRI (.nii file)", type=["nii", "gz"])
t1ce_file = st.file_uploader("Upload T1CE MRI (.nii file)", type=["nii", "gz"])
t2_file = st.file_uploader("Upload T2 MRI (.nii file)", type=["nii", "gz"])
seg_file = st.file_uploader("Upload SEG MRI (.nii file)", type=["nii", "gz"])


def load_and_display_nii(file, title, width=200):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name


        img = nib.load(tmp_path).get_fdata()
        os.remove(tmp_path)

        img=scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        img=img[56:184, 56:184, 13:141]
        mid_slice = 55  # take middle axial slice
        slice_img = img[:, :, mid_slice]
        

        st.write(f"**{title} - Middle Slice**")
        st.image(slice_img, caption=f"{title} - Slice {mid_slice}", clamp=True, width=width)


def show_array_streamlit(img, title="Image", width=200):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') 
    buf.seek(0)
    st.image(buf, caption=title, width=width)
    plt.close(fig)


def show_colored_mask(mask_array, title="Mask", width=200):
    """
    Displays a segmentation mask with colors using matplotlib's colormap.
    mask_array: 2D NumPy array (predicted class labels)
    """
    # Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
    # colormap = cm.get_cmap('viridis') # you can also try "jet", "tab10", etc.

    colormap = plt.get_cmap("viridis")
    colored_mask = colormap(mask_array / np.max(mask_array))[:, :, :3]  # remove alpha

    st.image((colored_mask * 255).astype(np.uint8), caption=title, width=width)





def combine_3(flair, t1ce, t2, temp_mask): 
    flair.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(flair.read())
        flair_path = tmp.name
    flair = nib.load(flair_path).get_fdata()
    os.remove(flair_path)

    t1ce.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(t1ce.read())
        t1ce_path = tmp.name
    t1ce = nib.load(t1ce_path).get_fdata()
    os.remove(t1ce_path)

    t2.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(t2.read())
        t2_path = tmp.name
    t2 = nib.load(t2_path).get_fdata()
    os.remove(t2_path) 

    temp_mask.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(temp_mask.read())
        seg_path = tmp.name
    temp_mask = nib.load(seg_path).get_fdata()
    os.remove(seg_path)


    flair=scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

    t1ce=scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

    t2=scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

    combine_3Images = np.stack([flair, t1ce, t2], axis=3)
    combine_3Images=combine_3Images[56:184, 56:184, 13:141]

    test_img = np.array(combine_3Images)

    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = my_model.predict(test_img_input)
    test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]

    # Mask work 
    # temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    # temp_mask= to_categorical(temp_mask, num_classes=4)


    col4, col5, col6 = st.columns(3) 

    with col4: 
        show_array_streamlit(combine_3Images[:, :, 55, 1], title="3 Images combined", width=200)
    with col5: 
        show_colored_mask(temp_mask[:, :, 55], title="Predicted Mask", width=200)
    with col6: 
        show_colored_mask(test_prediction_argmax[:, :, 55], title="Predicted Mask", width=200)




if flair_file and t1ce_file and t2_file and seg_file:
    st.subheader("📸 MRI Modalities - Slices")
    col1, col2, col3 = st.columns(3)
    with col1:
        load_and_display_nii(flair_file, "FLAIR", width=200) 
    with col2: 
        load_and_display_nii(t1ce_file, "T1CE", width=200) 
    with col3: 
        load_and_display_nii(t2_file, "T2", width=200)
    
    st.subheader("📸 Tumor Segmentation prediction")
    
    combine_3(flair_file, t1ce_file, t2_file, seg_file) 


