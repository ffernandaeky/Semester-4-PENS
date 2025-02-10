from pycaret.classification import *
import streamlit as st
import pandas as pd

model = load_model('obesity_pipeline')

def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    st.write(predictions_df)
    predictions = int(predictions_df['Label'].iloc[0]) 
    return predictions

def run():
    from PIL import Image
    image = Image.open('images/logo_sdt.png')
    image_diabetes = Image.open('images/stroke.jpg')

    st.sidebar.title('Praktikum Streamlit')
    st.sidebar.markdown("Aplikasi klasifikasi obesitas berdasarkan beberapa fitur dari data yang disediakan oleh pycaret")
    st.sidebar.info("Aplikasi ini contoh praktikum streamlit pada mata kuliah MLOps")
    st.sidebar.success("By : Eky Fernanda")
    st.sidebar.image(image)
    
    st.image(image_diabetes)
    st.title("Klasifikasi Tingkat Obesitas")
    st.markdown("Aplikasi ini bertujuan untuk memprediksi tingkat obesitas berdasarkan beberapa fitur.")
    pregnant = st.number_input('Jumlah kehamilan', min_value=0, max_value=20, value=2)
    plasma = st.number_input('Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral',min_value=0, max_value=300, value=90)
    bp = st.number_input('Tekanan darah diastolik (mm Hg)', min_value=0, max_value=200, value=80)
    ts = st.number_input('Tebal lipatan kulit trisep (mm)', min_value=0, max_value=150, value=0)
    serum = st.number_input('Insulin serum 2 jam (mu U/ml)', min_value=0, max_value=900, value=0)
    bmi = st.number_input('Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)', min_value=0.0, max_value=80.0, value=32.0, step=0.1)
    dp = st.number_input('Fungsi pedigri diabetes',min_value=0.000, max_value=3.000, value=0.258, step=0.001, format="%.3f")
    age = st.number_input('Usia (tahun)',min_value=0, max_value=100, value=22)
    
    input_dict = {'Number of times pregnant' : pregnant,
                 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test' : plasma,
                 'Diastolic blood pressure (mm Hg)' : bp,
                  'Triceps skin fold thickness (mm)' :ts ,
                  '2-Hour serum insulin (mu U/ml)' : serum,
                  'Body mass index (weight in kg/(height in m)^2)' : bmi,
                  'Diabetes pedigree function' : dp,
                  'Age (years)' : age,
                 }
    input_df = pd.DataFrame([input_dict])
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        obesity_mapping = {
            0: 'Insufficient Weight',
            1: 'Normal Weight',
            2: 'Obesity Type I',
            3: 'Obesity Type II',
            4: 'Obesity Type III',
            5: 'Overweight Level I',
            6: 'Overweight Level II'
        }
        predicted_obesity_level = obesity_mapping[output]
        st.write(f'Prediksi Tingkat Obesitas: {predicted_obesity_level}')
   
if __name__ == '__main__':
    run()
