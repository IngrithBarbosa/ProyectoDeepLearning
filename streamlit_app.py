import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

model = tf.keras.models.load_model(r'resources\cnn_model.h5')
label_map ={0: 'Benign', 1: 'Malignant'}

st.set_page_config(
    layout="centered", page_title="Classify image", page_icon="💊"
)

### Support functions
def generate_progress_bar(value):
    return f'<div style="width: 100%; border: 1px solid #eee; border-radius: 10px;"><div style="width: {value * 100}%; height: 24px; background: linear-gradient(90deg, rgba(62,149,205,1) 0%, rgba(90,200,250,1) 100%); border-radius: 10px;"></div></div>'


# UI
st.title('Clasificación de imágenes de mamografías')
# Texto introductorio
st.write("Suba una imagen de una mamografía que desee clasificar")
# Subir archivo 
uploaded_image = st.file_uploader("Subir imagen", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    image=image.resize((640, 640))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización

    # Realizar clasificación si hay datos cargados
    if st.button('Clasificar'):
        try:
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = label_map[predicted_class]
            
            st.subheader('Clasificación')
            st.write(f"Clasificación: {predicted_label}")

            #st.success("✅ Done!")

            #st.markdown(result_df.to_html(escape=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something happened: {e}", icon="🚨")
