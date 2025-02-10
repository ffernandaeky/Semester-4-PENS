from pycaret.classification import *
import streamlit as st
import pandas as pd
import plotly.express as px

# Load model yang sudah dilatih untuk klasifikasi
model = load_model('stroke_pipeline')

# Inisialisasi riwayat prediksi
prediction_history = []

# Definisikan fungsi klasifikasi untuk dipanggil
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    prediction = predictions_df.iloc[0, 1]  # Mengambil prediksi dari kolom 'Label' (misalnya 1 atau 0)
    return prediction

def plot_interactive_visualization(input_df, prediction):
    input_df['Prediction'] = prediction
    fig = px.scatter(input_df, x='age', y='avg_glucose_level', color='Prediction', 
                     title='Hubungan Usia dan Rata-rata Glukosa Darah',
                     labels={'age': 'Usia', 'avg_glucose_level': 'Rata-rata Glukosa Darah', 'Prediction': 'Prediksi Stroke'})
    st.plotly_chart(fig)

def run():
    # Gambar
    from PIL import Image
    image = Image.open('images/logo_sdt.png')
    image_stroke = Image.open('images/stroke.jpg')

    # Tambahkan sidebar ke aplikasi
    st.sidebar.image(image, use_column_width=True)
    st.sidebar.markdown(
        """
        <style>
        .css-1l02zno {
            text-align: center;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.sidebar.title('DEMO MLOPS STREAMLIT')

    # Tampilkan informasi di dalam card berwarna biru (Tentang Aplikasi)
    with st.sidebar.expander("Tentang Aplikasi"):
        st.markdown("""
            Aplikasi ini memprediksi kemungkinan seseorang mengalami stroke berdasarkan informasi pasien.
            Model ini dilatih menggunakan dataset Stroke Prediction dari Kaggle.
        """)

    # Tampilkan tombol expand/collapse untuk card "Tentang Model"
    with st.sidebar.expander("Tentang Model"):
        st.markdown("""
            Sebelum membuat model prediksi, dilakukan tahap eksplorasi data untuk memahami karakteristik dataset, 
            seperti distribusi variabel dan korelasi antar variabel. Fitur-fitur yang dipilih untuk dimasukkan 
            ke dalam model dipilih berdasarkan relevansi dengan masalah yang ingin diselesaikan. Setelah itu, 
            dilakukan pemrosesan data seperti penanganan missing values dan encoding variabel kategorikal. Model 
            yang dipilih untuk menyelesaikan masalah prediksi risiko stroke ini adalah model klasifikasi, yang 
            dievaluasi menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score. Model ini 
            telah dioptimalkan menggunakan teknik penyetelan parameter (hyperparameter tuning) untuk meningkatkan 
            performa dan generalisasi model. Hasil analisis menunjukkan bahwa faktor-faktor seperti usia dan BMI 
            mempengaruhi prediksi risiko stroke secara signifikan. Rekomendasi untuk penggunaan model ini termasuk 
            integrasi ke dalam praktik klinis untuk membantu identifikasi pasien dengan risiko tinggi dan mengambil 
            tindakan pencegahan yang sesuai.
        """)

    # Tambahkan card tutorial prediksi
    with st.sidebar.expander("Tutorial Melakukan Prediksi"):
        st.markdown("""
            Untuk melakukan prediksi, silakan lengkapi form di sebelah kanan dengan informasi pasien yang diperlukan.
            Isilah kolom-kolom seperti usia, keberadaan hipertensi, penyakit jantung, rata-rata glukosa darah, BMI,
            jenis kelamin, status pernikahan, jenis pekerjaan, tipe tempat tinggal, dan status merokok.
            Setelah itu, klik tombol 'Prediksi' untuk melihat hasil prediksi.
        """)

    st.sidebar.success("Dibuat oleh : Eky Fernanda")

    # Tambahkan judul dan subjudul ke antarmuka utama aplikasi
    st.image(image_stroke)
    st.title("KLASIFIKASI RISIKO STROKE")
    st.markdown("Aplikasi ini memprediksi risiko seseorang mengalami stroke berdasarkan beberapa fitur.")

    # Tambahkan input field untuk fitur-fitur dengan nilai default kosong
    gender = st.radio('Jenis Kelamin', ['Male', 'Female'], index=None)
    age = st.number_input('Usia', min_value=0, max_value=150, value=None)
    hypertension = st.radio('Hipertensi', ['Tidak', 'Ya'], index=None)
    heart_disease = st.radio('Penyakit Jantung', ['Tidak', 'Ya'], index=None)
    ever_married = st.radio('Pernah Menikah?', ['Yes', 'No'], index=None)
    work_type = st.selectbox('Jenis Pekerjaan', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], index=None)
    residence_type = st.radio('Tipe Tempat Tinggal', ['Urban', 'Rural'], index=None)
    avg_glucose_level = st.number_input('Rata-rata Glukosa Darah (mg/dL)', min_value=0.0, max_value=300.0, value=None)
    bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0, value=None)
    smoking_status = st.selectbox('Status Merokok', ['never smoked', 'formerly smoked', 'smokes'], index=None)

    # Persiapkan data input
    hypertension = 1 if hypertension == 'Ya' else 0
    heart_disease = 1 if heart_disease == 'Ya' else 0
    input_dict = {'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease,
                  'avg_glucose_level': avg_glucose_level, 'bmi': bmi, 'gender': gender,
                  'ever_married': ever_married, 'work_type': work_type,
                  'Residence_type': residence_type, 'smoking_status': smoking_status}  
    input_df = pd.DataFrame([input_dict])

    if st.button("Prediksi"):
        output = predict(model=model, input_df=input_df)
        prediction_history.append({'Input': input_dict, 'Prediction': output})  # Simpan riwayat prediksi

        st.subheader("Data Input:")
        st.write(input_df)  # Tampilkan tabel user

        # Keterangan Prediksi
        st.subheader("Keterangan Prediksi:")
        if output:
            st.error('Pasien **mengalami stroke**.')
            st.write("Pasien memiliki risiko stroke. Saran: Segera konsultasikan dengan dokter.")
        else:
            st.success('Pasien **tidak mengalami stroke**.')
            st.write("Pasien tidak memiliki risiko stroke. Tetap pertahankan gaya hidup sehat.")

        # Tampilkan visualisasi 
        st.subheader("Grafik Prediksi Stroke:")
        plot_interactive_visualization(input_df, output)

if __name__ == '__main__':
    run()
