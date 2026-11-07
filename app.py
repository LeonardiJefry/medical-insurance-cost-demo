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
    "chronic_count": (
        "Jumlah kondisi kronis",
        "Hitung kategori yang tercatat pada profil. Dataset mencakup hipertensi, diabetes, "
        "asma, PPOK, penyakit kardiovaskular, riwayat kanker, penyakit ginjal, penyakit hati, "
        "artritis, dan gangguan kesehatan mental.",
    ),
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
    "monthly_premium": (
        "Premi bulanan polis",
        "Gunakan premi polis yang sudah diketahui sebelum periode estimasi, "
        "dalam satuan biaya yang sama dengan dataset.",
    ),
    "deductible": (
        "Deductible pada polis",
        "Nilai biaya yang menjadi tanggungan pemegang polis sesuai ketentuan deductible. "
        "Gunakan satuan biaya yang sama dengan dataset.",
    ),
    "network_tier": (
        "Tingkat jaringan polis",
        "Pilih tingkat Bronze, Silver, Gold, atau Platinum yang tercatat pada polis sebelum periode estimasi.",
    ),
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
NUMERIC_DOMAINS = {
    "age": (18, 100),
    "bmi": (0.1, None),
    "chronic_count": (0, 10),
    "visits_last_year": (0, None),
    "hospitalizations_last_3yrs": (0, 1096),
    "days_hospitalized_last_3yrs": (0, 1096),
    "medication_count": (0, None),
    "monthly_premium": (0.0, None),
    "deductible": (0, None),
}


@st.cache_resource
def load_bundle(model_bytes):
    # Streamlit hashes the bytes, so replacing the artifact refreshes this cache.
    return pickle.loads(model_bytes)


def format_number(value, digits=2):
    return f"{value:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def render_field(name, spec):
    label, help_text = FIELD_TEXT[name]
    if spec["kind"] == "category":
        choices = list(spec["choices"])
        return st.radio(
            label,
            options=choices,
            index=choices.index(spec["default"]),
            format_func=lambda choice: CATEGORY_LABELS.get(choice, str(choice)),
            help=help_text,
            horizontal=True,
            key=name,
        )

    cast = int if spec["kind"] == "integer" else float
    domain_min, domain_max = NUMERIC_DOMAINS[name]
    minimum = spec.get("domain_min", domain_min)
    maximum = spec.get("domain_max", domain_max)
    return st.number_input(
        label,
        min_value=cast(minimum),
        max_value=cast(maximum) if maximum is not None else None,
        value=cast(spec["default"]),
        step=1 if cast is int else 0.1,
        help=help_text,
        key=name,
    )


def main():
    st.set_page_config(page_title="Estimasi Biaya Medis Berbasis Polis", page_icon="📊")
    st.title("Estimasi biaya medis berbasis polis")
    st.write(
        "Gunakan profil orang dewasa dan data polis yang diasumsikan sudah diketahui "
        "sebelum periode estimasi."
    )
    st.caption("Gunakan profil contoh untuk demo akademik. Asal klinis dataset belum terverifikasi.")

    try:
        bundle = load_bundle(MODEL_PATH.read_bytes())
    except FileNotFoundError:
        st.error(
            "Model belum tersedia. Jalankan notebook untuk menghasilkan berkas model, "
            "lalu muat ulang aplikasi."
        )
        st.stop()

    if bundle.get("schema_version", 1) not in (1, 2):
        st.error("Versi berkas model belum kompatibel dengan aplikasi ini.")
        st.stop()

    metrics = bundle["metrics"]
    r2_met = metrics["r2"] > 0.70
    mape_met = metrics["mape_pct"] < 15
    metrics_pass = r2_met and mape_met
    rubric_pass = bool(bundle.get("rubric_pass", metrics_pass)) and metrics_pass
    if rubric_pass:
        st.success("Ambang evaluasi tercapai pada benchmark dataset ini.")
    else:
        st.info("Kelulusan ambang proyek belum dikonfirmasi untuk model ini. Gunakan estimasi untuk simulasi.")

    fields = bundle["fields"]
    if "monthly_premium" in fields:
        st.caption("Hubungan matematis antara premi dan biaya pada dataset dibahas pada bagian evaluasi.")
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
            outside_training_range = []
            for field, value in inputs.items():
                spec = bundle["input_schema"][field]
                if spec["kind"] == "category":
                    continue
                training_min = spec.get("training_min", spec.get("min"))
                training_max = spec.get("training_max", spec.get("max"))
                if (
                    training_min is not None and value < training_min
                    or training_max is not None and value > training_max
                ):
                    outside_training_range.append(FIELD_TEXT[field][0])
            if outside_training_range:
                st.warning(
                    "Isian berikut berada di luar rentang data pelatihan sehingga estimasinya "
                    "memerlukan kehati-hatian. " + ", ".join(outside_training_range) + "."
                )
            row = pd.DataFrame([inputs], columns=fields)
            prediction = float(bundle["model"].predict(row)[0])
            if math.isfinite(prediction) and prediction >= 0:
                st.metric("Estimasi biaya medis tahunan", format_number(prediction))
                st.caption("Angka memakai satuan biaya pada dataset. Evaluasi berlaku untuk benchmark dataset ini.")
            else:
                st.error("Model tidak menghasilkan estimasi biaya yang valid untuk profil ini.")

    with st.expander("Cara membaca hasil dan evaluasi model"):
        st.write(f"Model yang digunakan **{bundle['model_name']}**.")
        if "model_id" in bundle:
            st.caption(f"Versi model {bundle['model_id']}")
        if "evaluation_role" in bundle:
            st.caption(f"Peran data evaluasi {bundle['evaluation_role']}")
        if len(fields) == 10:
            st.write(
                "Sepuluh masukan dipertahankan sesuai rancangan proyek. "
                "Analisis lima fitur penting membantu menjelaskan prediksi model."
            )
            st.write(
                "Peringkat tetap menampilkan lima fitur meskipun kekuatan prediksi dapat "
                "terkonsentrasi pada premi, deductible, dan tingkat jaringan polis."
            )
        st.write(
            "Skor berikut berasal dari benchmark pada dataset yang sama. "
            "Data polis diasumsikan sudah tersedia sebelum periode estimasi."
        )
        r2_status = "Tercapai" if r2_met else "Belum tercapai"
        mape_status = "Tercapai" if mape_met else "Belum tercapai"
        st.table(
            {
                "Metrik benchmark data uji": ["R²", "MAE", "MAPE", "WAPE"],
                "Nilai": [
                    format_number(metrics["r2"], 6),
                    f"{format_number(metrics['mae'])} satuan biaya",
                    f"{format_number(metrics['mape_pct'], 4)}%",
                    f"{format_number(metrics['wmape_pct'], 4)}%",
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
        if "monthly_premium" in fields:
            st.write(
                "Audit notebook menemukan bahwa premi tahunan pada dataset mengikuti persamaan "
                "berikut hingga galat pembulatan. Premi bulanan mendekati premi tahunan dibagi 12."
            )
            st.code(
                "annual_premium = 200 + 0.01 × deductible + tier_factor × annual_medical_cost\n"
                "tier_factor = Bronze 0.096 | Silver 0.120 | Gold 0.144 | Platinum 0.174",
                language=None,
            )
            st.write(
                "Hubungan ini membuat biaya dan premi saling terkait secara matematis pada dataset. "
                "Skor benchmark dipengaruhi struktur tersebut. Kemampuan memperkirakan biaya "
                "periode mendatang perlu diuji dengan data polis dan biaya dari periode terpisah."
            )
        st.markdown(f"[Dataset Medical Insurance Cost Prediction di Kaggle]({DATASET_URL})")


if __name__ == "__main__":
    main()
