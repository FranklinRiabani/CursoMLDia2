import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# configurar la pagina
st.set_page_config(page_title="Proyecto ML - UPDS - Regresion Lineal", page_icon=":bar_chart:", layout="wide")

st.title("Aprendizaje Supervisado - Regresion Lineal")


@st._cache_data
def cargar_datos(archivo):
    if archivo:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith('.xls'):
            df=pd.read_excel(archivo)
        else:
            raise ValueError('Formato de archivo no valido, solo se leen csv y xls')
        return df
    else:
        return None



# cargar datos
archivo = st.sidebar.file_uploader("Cargar archivo", type=["csv", "XLS"])

if archivo is not None:
    df=cargar_datos(archivo)

    if df is not None:
        st.session_state.df=df

        st.markdown('### Grafico de Dispersión')

        fig=plt.figure(figsize=(6,6))
        plt.title("Regresion Lineal")
        plt.scatter(df['horas'],df['ingreso'],color='blue',marker='o',s=50)
        plt.xlabel('Horas de trabajo')
        plt.ylabel('Ingresos')

        st.pyplot(fig)

        st.markdown('### Regresion Lineal')
        from sklearn.linear_model import LinearRegression

        horas=df['horas'].values.reshape(-1,1)
        ingreso=df['ingreso'].values.reshape(-1,1)

        regresion = LinearRegression()

        modelo=regresion.fit(horas,df['ingreso'])

        st.write(' Coeficiente de la regresion (m):',modelo.coef_)

        st.write('Intercepto de la regresion (b):',modelo.intercept_)

        entrada = [[41.7],[40.5],[39.6],[44]]

        modelo.predict(entrada)

        st.write('Ingreso para 41.7 horas de trabajo:',modelo.predict(entrada))

        # grafico de la regresion

        fig=plt.figure(figsize=(6,6))

        plt.scatter(df['horas'],df['ingreso'],color='blue',marker='o',s=50)

        plt.plot(entrada,modelo.predict(entrada),color='pink')
        
        plt.scatter(entrada,modelo.predict(entrada),color='red')


        plt.xlabel('Horas de trabajo')
        plt.ylabel('Ingresos')

        st.pyplot(fig)


