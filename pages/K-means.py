import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# configurar la pagina
st.set_page_config(page_title="Proyecto ML - UPDS - Kmeans", page_icon=":bar_chart:", layout="wide")

st.title("Aprendizaje no supervisado - Kmeans")

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

archivo = st.sidebar.file_uploader("Cargar archivo", type=["csv", "XLS"])

if archivo is not None:
    df=cargar_datos(archivo)

    if df is not None:
        st.session_state.df=df

        st.markdown('### Normalizacion de Datos')

        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler

        scarler=MinMaxScaler().fit(df.values)

        datos_normalizados=pd.DataFrame(scarler.transform(df.values),columns=['Antiguedad','Precio'])

        st.write(datos_normalizados.head())

        # seleccionar el numero de clusters
        from sklearn.cluster import KMeans

        k=st.sidebar.slider('Numero de Clusters',2,10,2)

        # numero de iteraciones
        iteraciones=st.sidebar.slider('Numero de Iteraciones',100,1000,100)

        # agregar ramdom state
        random_state=st.sidebar.slider('Random State',0,100,0)

        # crear el modelo
        modelo=KMeans(n_clusters=k,max_iter=iteraciones,random_state=random_state).fit(datos_normalizados)

        # calcular el coeficiente de silueta
        from sklearn.metrics import silhouette_score

        coeficiente=silhouette_score(datos_normalizados,modelo.labels_)

        col1,col2=st.columns(2)

        with col1:
            fig=plt.figure(figsize=(6,6))
            plt.scatter(datos_normalizados['Antiguedad'],datos_normalizados['Precio'],c=modelo.labels_,s=150)
            plt.scatter(modelo.cluster_centers_[:,0],modelo.cluster_centers_[:,1],c='red',s=250,marker='x')
            plt.title('Kmeans')
            plt.xlabel('Antiguedad')
            plt.ylabel('Precio')
            st.pyplot(fig)
        with col2:
            st.markdown('### Medidad de Evaluacion')
            st.write('Numero de Clusters:',k)
            st.write('Coeficiente de Silueta:',coeficiente)
            st.write('Inercia:',modelo.inertia_)

        



        