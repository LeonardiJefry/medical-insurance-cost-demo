import math
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "insurance_model.pkl"
DATASET_URL = "https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction"
FIELD_TEXT = {
    "age": ("Usia (tahun)", "Usia pada awal periode yang ingin diperkirakan."),
    "bmi": ("Indeks massa tubuh (kg/m²)", "Gunakan berat (kg) dibagi tinggi (m) kuadrat."),
    "smoker": ("Status merokok", "Pilih status sebelum periode perkiraan."),
    "chronic_count": ("Jumlah kondisi kronis", "Hitung kondisi kronis yang sudah diketahui."),
    "visits_last_year": (
        "Jumlah kunjungan medis dalam 1 tahun terakhir",
        "Hitung kunjungan sebelum periode perkiraan.",
    ),
    "hospitalizations_last_3yrs": (
        "Jumlah rawat inap dalam 3 tahun terakhir",
        "Hitung jumlah kejadian rawat inap.",
    ),
    "days_hospitalized_last_3yrs": (
        "Total hari rawat inap dalam 3 tahun terakhir",
        "Jumlahkan lama seluruh rawat inap dalam periode tersebut.",
    ),
    "medication_count": ("Jumlah obat", "Jumlah obat yang tercatat sebelum periode perkiraan."),
    "region": ("Wilayah tempat tinggal", "Wilayah geografis sesuai kategori pada dataset."),
    "urban_rural": ("Tipe kawasan tempat tinggal", "Pilih kategori lingkungan tempat tinggal."),
}
CATEGORY_LABELS = {
    "Never": "Tidak pernah merokok",
    "Former": "Pernah merokok, sudah berhenti",
    "Current": "Saat ini merokok",
    "North": "Utara",
    "East": "Timur",
    "Central": "Tengah",
    "South": "Selatan",
    "West": "Barat",
    "Urban": "Perkotaan",
    "Suburban": "Pinggiran kota",
    "Rural": "Perdesaan",
}


@st.cache_resource
def load_bundle():
    # This path points to the model exported by the accompanying notebook.
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def format_number(value, digits=2):
    return f"{value:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def render_field(name, spec):
    label, help_text = FIELD_TEXT[name]
    if spec["kind"] == "category":
        choices = list(spec["choices"])
        return st.selectbox(
            label,
            options=choices,
            index=choices.index(spec["default"]),
            format_func=lambda choice: CATEGORY_LABELS.get(choice, choice),
            help=help_text,
            key=name,
        )

    cast = int if spec["kind"] == "integer" else float
    return st.number_input(
        label,
        min_value=cast(spec["min"]),
        max_value=cast(spec["max"]),
        value=cast(spec["default"]),
        step=1 if cast is int else 0.1,
        help=help_text,
        key=name,
    )


def main():
    st.set_page_config(page_title="Estimasi Biaya Medis Tahunan", page_icon="📊")
    st.title("Estimasi biaya medis tahunan")
    st.write("Simulasikan total biaya medis satu tahun ke depan untuk orang dewasa (usia 18 tahun ke atas).")
    st.caption("Gunakan profil contoh untuk demo akademik. Asal klinis dataset belum terverifikasi.")

    try:
        bundle = load_bundle()
    except FileNotFoundError:
        st.error(
            "Model belum tersedia. Jalankan notebook untuk menghasilkan berkas model, "
            "lalu muat ulang aplikasi."
        )
        st.stop()

    metrics = bundle["metrics"]
    r2_met = metrics["r2"] > 0.70
    mape_met = metrics["mape_pct"] < 15
    if not (r2_met and mape_met):
        st.info("Hasil uji model masih di luar ambang tugas. Gunakan estimasi untuk simulasi.")

    fields = bundle["fields"]
    st.subheader("Profil untuk simulasi")
    st.write(
        f"Isi {len(fields)} kolom yang digunakan model terpilih. "
        "Nilai awal bisa langsung dipakai sebagai contoh."
    )

    with st.form("prediction_form"):
        inputs = {}
        for field in fields:
            inputs[field] = render_field(field, bundle["input_schema"][field])
        submitted = st.form_submit_button("Hitung estimasi", type="primary")

    if submitted:
        stays = inputs.get("hospitalizations_last_3yrs")
        days = inputs.get("days_hospitalized_last_3yrs")
        if stays is not None and days is not None and (days < stays or (stays == 0 and days > 0)):
            st.error(
                "Isi 0 hari jika tidak pernah rawat inap. "
                "Jika pernah, isi minimal 1 hari untuk tiap kejadian."
            )
        else:
            row = pd.DataFrame([inputs], columns=fields)
            prediction = float(bundle["model"].predict(row)[0])
            if math.isfinite(prediction) and prediction >= 0:
                st.metric("Perkiraan biaya medis per tahun", format_number(prediction))
                st.caption("Angka ditampilkan dalam satuan biaya pada dataset dan digunakan untuk simulasi.")
            else:
                st.error("Model tidak menghasilkan estimasi biaya yang valid untuk profil ini.")

    with st.expander("Cara membaca hasil dan evaluasi model"):
        st.write(f"Model yang digunakan **{bundle['model_name']}**.")
        st.write(
            "Notebook membandingkan model dengan 10 masukan awal dan versi ringkas. "
            "Formulir mengikuti masukan yang digunakan oleh model terpilih."
        )
        r2_status = "Tercapai" if r2_met else "Belum tercapai"
        mape_status = "Tercapai" if mape_met else "Belum tercapai"
        st.table(
            {
                "Metrik data uji": ["R²", "MAE", "MAPE", "WAPE"],
                "Nilai": [
                    format_number(metrics["r2"], 3),
                    f"{format_number(metrics['mae'])} satuan biaya",
                    f"{format_number(metrics['mape_pct'])}%",
                    f"{format_number(metrics['wmape_pct'])}%",
                ],
                "Target tugas": [
                    f"Lebih dari 0,70 · {r2_status}",
                    "—",
                    f"Di bawah 15% · {mape_status}",
                    "—",
                ],
            }
        )
        st.caption(
            "R² menilai seberapa baik model menjelaskan variasi biaya. MAE adalah selisih absolut rata-rata. "
            "MAPE merata-ratakan persentase selisih absolut per profil. WAPE membandingkan total selisih "
            "absolut dengan total biaya aktual. Ketepatan untuk tiap profil dapat berbeda."
        )
        st.markdown(f"[Dataset Medical Insurance Cost Prediction di Kaggle]({DATASET_URL})")


if __name__ == "__main__":
    main()
