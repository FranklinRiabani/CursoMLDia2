import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# configurar la pagina
st.set_page_config(page_title="Proyecto ML - UPDS - Analisis Exploratorio", page_icon=":bar_chart:", layout="wide")

# titulo
st.title("Analisis Exploratorio de Datos")  

opcion = st.sidebar.selectbox('Seleccione una Opcion',['Analisis Exploratorio','Correlacion de Variables','Normalizacion de Datos'])

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


if opcion == 'Analisis Exploratorio':
    # cargar datos
    archivo = st.sidebar.file_uploader("Cargar archivo", type=["csv", "XLS"])

    if archivo is not None:
        df=cargar_datos(archivo)

        st.session_state.df=df
        st.write(df.head())

        st.write('### Estadisticas Descriptivas')
        st.write(df.describe())

        st.write('### Informacion del Dataset')
        st.write(df.info())

        st.write('### Valores Nulos')
        st.write(df.isnull().sum())

        # graficos
        st.write('### Graficos')

        fig=plt.figure()
        sns.pairplot(df)
        st.pyplot(fig)

        st.write(sns.pairplot(df))

elif opcion == 'Correlacion de Variables':
    st.markdown('## Correlacion de Variables')

    if 'df' not in st.session_state:
        st.write('No hay datos cargados')
    else:
        df=st.session_state.df
        st.markdown('### Matriz de Correlacion')
        st.write(df.corr())
        fig=plt.figure()
        sns.heatmap(df.corr(),annot=True)
        st.pyplot(fig)


elif opcion == 'Normalizacion de Datos':
    st.markdown('## Normalizacion de Datos')
    if 'df' not in st.session_state:
        st.write('No hay datos cargados')
    else:
        df=st.session_state.df

        st.markdown('### Normalizacion de Datos por Maximo y Minimo')
        from sklearn.preprocessing import MinMaxScaler

        min_max_scaler = MinMaxScaler()
        df_norm_minmax = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)

        st.markdown('### Normalizacion de Datos por Standarizacion')
        from sklearn.preprocessing import StandardScaler

        std_scaler = StandardScaler()
        df_norm_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)

        # graficos

        fig=plt.figure(figsize=(10,5))
        ax1=fig.add_subplot(1,2,1)
        ax2=fig.add_subplot(1,2,2)

        ax1.set_title('Datos Originales')
        ax1.scatter(df['ingreso'],df['horas'])
        ax1.set_xlabel('Ingreso')
        ax1.set_ylabel('Horas')

        ax2.set_title('Datos Normalizados')
        ax2.scatter(df_norm_std['ingreso'],df_norm_std['horas'])
        ax2.set_xlabel('Ingreso')
        ax2.set_ylabel('Horas')

        st.pyplot(fig)
