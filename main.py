from dataclasses import replace
from re import A
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from tensorflow.keras.models import load_model
import pickle 
import base64
import textwrap
import streamlit.components.v1 as components
from sklearn.preprocessing import PolynomialFeatures
import os


st.set_page_config(page_title="Impact ML", layout="wide")

#Allow to insert a CSS file to combine with python scripting
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<div style="background-color: white;"> <img src="data:image/svg+xml;base64,%s"/></div>' % b64
    st.write(html, unsafe_allow_html=True)


#load the linear model

pkl_filename = (r"Models\Polynomial.pkl")
file = open(os.path.join(pkl_filename, '.pkl'))
pol = pickle.load(file)

#Load the NN
NN = load_model(r"Models\PO(Normal).h5")
def apportion(data,forecast):    
    pv_total = np.sum(data["Sales"])
    data[["Sales","PO"]] = data[["Sales","PO"]].astype(float)
    data["Sales"] = data["Sales"].apply(lambda x: x/pv_total*forecast)    
    return data

def polyl(x):
    #Se define el grado del polinomio
    poli_reg = PolynomialFeatures(degree = 2)
    #Se transforma las características existentes en características de mayor grado
    y = poli_reg.fit_transform(x)    
    return y

def NN_prediction(data,i):  
    predictions = []                    
    #Evaluate the month, if is January assign 0 else use the model
    if i[1][0] == 1:
        current_pred = 0
    else:        
    #Predict the value for each element in the x_train list     
        current_pred = NN.predict(np.asarray(i[1]).astype(np.float32).reshape(1,-1))            
    #Append the results    
    predictions.append(float(np.array(current_pred)))
    return predictions
    

#Load the stylesheet
#local_css("style\style.css")
path = r"Databases\Templates.xlsx"

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Selecciona la opcion que deseas",
    ("Inicio", "Predicciones", "Metricas")
)
if add_selectbox == "Inicio":      
    st.title("Bienvenido al predictor de Impactos")

    st.write("""
           ### ¡Conoce más sobre como funciona este impacto!
           Este impacto es generado de manera aleatoria a partir de errores humanos por lo que no existe una manera concreta de poder estimar y preveer este impacto.                                 
        """)
    st.write("Para poder obtener una estimación sobre este impacto se opto por desarrollar una red neuronal simple que recibe como entrada 2 parametros:")
    st.markdown("-Mes")
    st.markdown("-Venta")
    
    st.write("La grafica siguiente muestra como se encuentran distribuidos nuestros datos:")
    
    #### PLOT DATASET ####
    
    #We extract only the data that we need from the Dataset and create a new file for python
    forecast_df = pd.read_csv(r"Databases\PO-DataFrame.csv",delimiter="|")
    #Clean blank spaces from columns with strip
    forecast_df.rename(columns=lambda x: x.strip(),inplace=True)
    #Convert to number the month
    forecast_df["Month"] = forecast_df["Month"].apply(lambda x: dt.datetime.strptime(x[:3], "%b").month)
    #Rename the column
    forecast_df.rename(columns={"Total Net Sales":"Actual","PO-Total":"PO"},inplace=True)
    forecast_df.drop(columns={"Product Format","Year"},axis=0,inplace=True)
    
    import matplotlib as mpl
    mpl.style.use('seaborn')
    X_p = np.asarray(forecast_df["Actual"]).astype(np.float32).reshape(-1,1)
    y_p = np.asarray(forecast_df["PO"]).astype(np.float32).reshape(-1,1)    

    fig = plt.figure(figsize=(10,6))    
    plt.scatter(X_p, y_p, alpha=0.8,cmap='white-grid')

    plt.xlabel('Ventas')
    plt.ylabel('Impacto')    

    #plt.show()
    st.pyplot(fig)
    
    st.write("La red neuronal otorga como resultado el impacto a nivel total de dicho mes y valores introducidos.")       
        
    st.write("Si quieres concer más sobre su entrenamiento y arquitectura da clic abajo!")

    with st.expander("Explicación"):
        st.write("Esta red neuronal fue entrenada simulando un periodo de tiempo de Enero-2020 a Febrero 2022.")
        st.write("El problema principal al desarrollar esta red neuronal es la poca cantidad de datos para entrenar el modelo y encontrar una relación más concreta.")
        st.write("Para obtener mayor cantidad de datos y obtener mayor precisión en nuestro modelo la información original fue desglozada a un nivel más especifico realizando un prorrateo de la venta y el impacto de cada mes en todos los productos vendidos en dicho periodo.")
        st.write("")
        st.write("Para las predicciones la información general con la que se alimenta es transformada y agrupada en distintos niveles con plantillas previamente desarrolladas y este valor es pasado a nuestra red neuronal para estimar el impacto")        
        
        st.write("""La red neuronal cuenta con una arquitectura basica con multiples capas de tipo "Dense" ocultas con alrededor de 1,000 neuronas distribuidas.""")                        
        st.write("La siguiente ilustración representa de manera gráfica una red neuronal simple similar a la que utilizamos pero con menor numero de capas y neuronas.")
        f = open("image.svg","r",encoding="utf-8")
        lines = f.readlines()
        line_string=''.join(lines)    
        components.html(render_svg(line_string))                
        
    st.warning("La informacion y el modelo original fue desarrollado para una empresa particular, el modelo y la información presentada imitan un comportamiento similar sin embargo los resultados presentados no representan ningún valor real.")    
        
elif add_selectbox == "Predicciones":
    st.title("¡Comencemos a predecir!")    
    st.write("""
        ### Una vez que entendemos como funciona, comencemos a realizar predicciones con el modelo de machine learning
         """)
    st.write("Por favor introduce la información correspondiente para el modelo..")    

    amount = str(st.text_input("Introduce el monto de venta."))
    year = str(st.text_input("Introduce el año a predecir."))
    month = st.selectbox("Selecciona el mes",("Month","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"))                                
    
    if month == "Month":
        pass
    else:
        #Load the DataFrame of the Month Selected
        #Define the limit of rows to show            
        limit = 50
        #Divide the forecast of the month in the average of proportion that represnts the GM Allied Channel and Source        
        amount = float(amount) 
        st.write("La infromación introducida sera utilizada para realizar las predicciones.")            
        st.write("")
        
        df = pd.read_excel(path,sheet_name=str(month).upper())
        headers = ["Month","Day","PN","Sales","PO"]
        df.columns = headers        
                
        new_df = apportion(df,amount)              
                
        st.write("Descomponemos y reagrupamos la información obteniendo el siguiente DataFrame con nuestras predicciones:")    
        
        #Convert to number the month
        new_df["Month"] = new_df["Month"].apply(lambda x: dt.datetime.strptime(x[:3], "%b").month)
        #Rename the column
        new_df.rename(columns={"Sales":"Actual"},inplace=True)
        #Obtain the Date Format
        new_df["Date"] = ""
        #Assign per day                
        new_df["Date"] = new_df["Day"].apply(str)+"/"+new_df["Month"].apply(str)+"/"+year 
        new_df["Date"] = new_df["Date"].apply(lambda x: dt.datetime.strptime(x,'%d/%m/%Y'))    
        #Drop the original date columns
        new_df.drop(columns={"Month","Day"},axis=1,inplace=True)
        #Transforming to the final shape
        final_df = new_df.pivot_table(['PO','Actual'], 'Date', aggfunc=np.sum)
        final_df = final_df.reset_index()
        final_df["Month"] = final_df["Date"].apply(lambda x: x.month)    
        final_df["Date"] = pd.to_datetime(final_df["Date"])
        final_df = final_df.set_index('Date').asfreq('D')
        
        #Start prediction with NN
        x_NN = final_df[["Month","Actual"]].values                
        x_NN = np.array(x_NN).astype(np.float32)
        predictions_NN = []
        for i in enumerate(x_NN):
            predictions_NN.append(NN_prediction(x_NN,i))
    
        PO_df = final_df.drop("Month",axis=1)
        PO_df["PO"] = predictions_NN
        PO_df["PO"] = PO_df["PO"].apply(str)
        PO_df["PO"] = PO_df["PO"].str.replace("[","")
        PO_df["PO"] = PO_df["PO"].str.replace("]","")
        PO_df["PO"] = PO_df["PO"].astype(float)
    
        #Start prediction with linear regression
        if month == "Jan":
            PO_df["Polynomial"] = 0
        elif month == "Anual":
            x = final_df[["Actual"]].values            
            x = np.array(x).astype(np.float32)
            x_pol = polyl(x)            
            prediction_pol = pol.predict(x_pol)
            PO_df["Polynomial"] = prediction_pol
        else:
            x = final_df[["Actual"]].values                
            x = np.array(x).astype(np.float32)
            x_pol = polyl(x)            
            prediction_pol = pol.predict(x_pol)
            PO_df["Polynomial"] = prediction_pol
    
        #Average of both predictions
        PO_df["Profit Out"] = (PO_df["PO"] + PO_df["Polynomial"])/2        
        PO = np.sum(PO_df["PO"])
        LIN = np.sum(PO_df["Polynomial"])
        AVG = '$' + (np.sum(PO_df["Profit Out"])/1000000).round(4).astype(str) + ' M'
        
        res = PO_df.drop(columns={"PO","Polynomial"},axis=1) 
        res = res.rename({"Actual":"Ventas por periodo", "Profit Out":"Predicción Impacto Generado"},axis=1)       
        res
        
        total = np.sum(PO_df["Actual"])
        total = '$' + (total/1000000).round(4).astype(str) + ' M'
       
        
        #Apply CSS formats
        text = "<br><div>Venta Total: <span class='highlight blue'>   " 
        text2 = (str(total))
        text3 = "</span></div><br>"
        text = text + text2 + text3
        st.write(text, unsafe_allow_html=True)
        
        text = "<div>Impacto Total: <span class='highlight blue'>   " 
        text2 = (str(AVG))
        text3 = "</span></div>"
        text = text + text2 + text3
        st.write(text, unsafe_allow_html=True)
        
elif add_selectbox == "Metricas":
    st.title("¡Evaluemos el modelo!")
    st.write("""
        ### Comparemos las predicciones con la información de entrenamiento
        A continuación puedes comparar la precisión del modelo por mes
         """)
    
    metrics_df = pd.read_excel(r"Databases\PO-Dataset.xlsx",sheet_name="Metrics")
    
    metrics_df[["Year","Month"]] = metrics_df[["Year","Month"]].astype(str)
    year = str(st.selectbox("Selecciona el año ",("2020","2021","2022")))
    
    if year == "2022":
        month = st.selectbox("Selecciona el mes",("Jan","Feb"))                                
    else:
        month = st.selectbox("Selecciona el mes ",("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"))                                
    
    df_filter = metrics_df.where(metrics_df["Year"] == year)
    df_filter = df_filter.where(df_filter["Month"] == month)
    df_filter.dropna(inplace=True)
    
    df = pd.read_excel(path,sheet_name=str(month).upper())
    headers = ["Month","Day","PN","Sales","PO"]
    df.columns = headers        
                
    new_df = apportion(df,np.sum(df_filter["Total Net Sales"]))                                      
    #Convert to number the month
    new_df["Month"] = new_df["Month"].apply(lambda x: dt.datetime.strptime(x[:3], "%b").month)
    #Rename the column
    new_df.rename(columns={"Sales":"Actual"},inplace=True)
    #Obtain the Date Format
    new_df["Date"] = ""
    #Assign per day                
    new_df["Date"] = new_df["Day"].apply(str)+"/"+new_df["Month"].apply(str)+"/"+year 
    new_df["Date"] = new_df["Date"].apply(lambda x: dt.datetime.strptime(x,'%d/%m/%Y'))    
    #Drop the original date columns
    new_df.drop(columns={"Month","Day"},axis=1,inplace=True)
    #Transforming to the final shape
    final_df = new_df.pivot_table(['PO','Actual'], 'Date', aggfunc=np.sum)
    final_df = final_df.reset_index()
    final_df["Month"] = final_df["Date"].apply(lambda x: x.month)    
    final_df["Date"] = pd.to_datetime(final_df["Date"])
    final_df = final_df.set_index('Date').asfreq('D')
        
    #Start prediction with NN
    x_NN = final_df[["Month","Actual"]].values                
    x_NN = np.array(x_NN).astype(np.float32)
    predictions_NN = []
    for i in enumerate(x_NN):
        predictions_NN.append(NN_prediction(x_NN,i))
    
    PO_df = final_df.drop("Month",axis=1)
    PO_df["PO"] = predictions_NN
    PO_df["PO"] = PO_df["PO"].apply(str)
    PO_df["PO"] = PO_df["PO"].str.replace("[","")
    PO_df["PO"] = PO_df["PO"].str.replace("]","")
    PO_df["PO"] = PO_df["PO"].astype(float)
    
    #Start prediction with polynomial regression
    if month == "Jan":
        PO_df["Polynomial"] = 0
    else:
        x = final_df[["Actual"]].values                
        x = np.array(x).astype(np.float32)
        x_pol = polyl(x)            
        prediction_pol = pol.predict(x_pol)
        PO_df["Polynomial"] = prediction_pol
          
    #Average of both predictions
    PO_df["Profit Out"] = (PO_df["PO"] + PO_df["Polynomial"])/2        
    PO = np.sum(PO_df["PO"])
    LIN = np.sum(PO_df["Polynomial"])    
    
    df_filter["Predicted"] = (PO + LIN) / 2
    df_filter.rename({"PO-Impact":"Impacto Real"},axis=1,inplace=True)    
    df_filter
            
    
    real_total = np.sum(df_filter["Impacto Real"])
    pred_total = np.sum(df_filter["Predicted"])
    diff = np.abs(real_total - pred_total)
    
    if pred_total > real_total:
        accuracy = real_total / pred_total       
    else:
        accuracy = pred_total / real_total
                   
    accuracy = float(accuracy) * 100
    
    diff = str(diff).replace("[","")
    diff = diff.replace("]","")
    diff = "$" + str(float(diff)/1000000)
    
    text = "<div>La diferencia del mes seleccionado es: <br> <span class='highlight blue'>   "
    text2 = str(diff)[:5] + "M"
    text3 = "</span></div>"
    text = text + text2 + text3
    st.write(text, unsafe_allow_html=True)
    
    text = "<br><div>La precisión del mes seleccionado es: <br> <span class='highlight blue'>   "
    text2 = str(accuracy)[:5] + "%"
    text3 = "</span></div>"
    text = text + text2 + text3
    st.write(text, unsafe_allow_html=True)
    
else:
    pass
    
        
    

    

    
                