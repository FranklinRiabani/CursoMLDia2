import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

st.subheader("Tokenizacion")
texto = "Hola ¿como estas?"

tokeks = texto.split()

st.write(tokeks)

# colocar todo en minusculas

texto = texto.lower()

st.write(texto)

tokens = texto.split()

st.write(tokens)


texto = "El gato es negro  y el perro es blanco"


# remover los stops words

import nltk


nltk.download('stopwords')



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('spanish'))

st.write(stop_words)


texto=texto.lower()

tokens = word_tokenize(texto)

st.write(tokens)

texto_filtrado=[word for word in tokens if not word in stop_words]

st.write(texto_filtrado)

st.subheader('stemming y lematizacion')

# descargar  nltk.download('wordnet')

nltk.download('wordnet')

from nltk.stem import SnowballStemmer


# crear el stemmer en espanol
stemmer = SnowballStemmer('spanish')

# vamos a probar como funciona

st.write(stemmer.stem('caminando'))
st.write(stemmer.stem('caminar'))
st.write(stemmer.stem('camino'))






