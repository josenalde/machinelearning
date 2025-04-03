# data app for iris prediction (classification)
import streamlit as st
from keras.models import load_model
import numpy as np

model = load_model('model.keras')

# assumindo este labelEncoder
class_labels = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

# st.set_page_config(layout="wide")

st.title('Classificação de Flor :blue[Iris]')

petal_length_input = st.slider("Comprimento da pétala", 1.0, 6.9, (1.0+6.9)/2)
petal_width_input = st.slider("Espessura da pétala", 0.1, 2.5, (0.1+2.5)/2)

btn = st.button('Pergunte à IA')

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
.answer-color {
    color:yellow;
}
</style>
""", unsafe_allow_html=True)

# st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)

if btn:
    classes_prob = model.predict(np.array(
        [[petal_length_input, petal_width_input]]))  # cuidado com a sequencia
    # print(classes_prob) #devolve array com as probabilidades de cada classe (multiclasse)
    st.html(
        f"""<p class="answer-color">{class_labels[np.argmax(classes_prob)]}</p>""")
    # st.subheader(class_labels[np.argmax(classes_prob)])
    if (np.argmax(classes_prob) == 0):
        st.image('img/setosa.jpg')
    elif (np.argmax(classes_prob) == 1):
        st.image('img/versicolor.jpg')
    else:
        st.image('img/virginica.jpg')

# some_number = st.number_input('Enter a number')
