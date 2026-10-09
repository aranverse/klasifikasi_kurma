import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Klasifikasi Kematangan Kurma",
	page_icon = "🌴"
)

model = joblib.load("model_klasifikasi_kurma")

st.title("🌴 Klasifikasi Kematangan Kurma")
st.markdown("Aplikasi Machine Learning Classification untuk Memprediksi Kurma (Khalal / Rutab / Tamr)")

ukuran_cm = st.slider("Ukuran (cm)", 1.0, 6.0, 3.5)
berat_g = st.slider("Berat (g)", 1.0, 25.0, 10.0)
kekerasan_N = st.slider("Kekerasan (N)", 0.0, 100.0, 40.0)
kadar_gula_brix = st.slider("Kadar Gula (Brix)", 0.0, 90.0, 40.0)
kadar_air_pct = st.slider("Kadar Air (%)", 0.0, 100.0, 40.0)

varietas = st.pills("Varietas", ["Ajwa", "Deglet Noor", "Medjool", "Zahidi"], default="Ajwa")
tingkat_warna = st.pills("Tingkat Warna", ["Cokelat Muda", "Cokelat Tua", "Hitam", "Kuning"], default="Cokelat Muda")
grade_cacat = st.pills("Grade Cacat", ["Tidak Ada", "Ringan", "Sedang", "Berat"], default="Tidak Ada")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[
		ukuran_cm, berat_g, kekerasan_N, kadar_gula_brix, kadar_air_pct,
		varietas, tingkat_warna, grade_cacat
	]], columns=[
		"ukuran_cm", "berat_g", "kekerasan_N", "kadar_gula_brix", "kadar_air_pct",
		"varietas", "tingkat_warna", "grade_cacat"
	])

	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])

	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan ❤️ oleh Rofiqoh Dini Maharani")