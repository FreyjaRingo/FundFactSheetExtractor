import streamlit as st
import pandas as pd
import json
import time
import io
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import pdfplumber
from groq import Groq
from supabase import create_client, Client
import google.generativeai as genai

# ==========================================
# 1. INISIALISASI INFRASTRUKTUR
# ==========================================
@st.cache_resource
def init_supabase():
    return create_client(st.secrets["supabase"]["URL"], st.secrets["supabase"]["KEY"])

def init_gemini():
    genai.configure(api_key=st.secrets["gemini"]["API_KEY"])
    return genai.GenerativeModel('gemini-2.5-flash')

def init_gemini_2():
    try:
        api_key_2 = st.secrets["gemini"]["API_KEY_2"]
    except KeyError:
        raise ValueError("API_KEY_2 untuk Gemini belum terbaca. Silakan RESTART server Streamlit Anda.")
    
    if not api_key_2:
        raise ValueError("API_KEY_2 untuk Gemini kosong di secrets.toml")
    
    genai.configure(api_key=api_key_2)
    return genai.GenerativeModel('gemini-2.5-flash')

def init_groq():
    return Groq(api_key=st.secrets["groq"]["API_KEY"])

def init_groq_fallback():
    try:
        api_key_2 = st.secrets["groq"]["API_KEY_2"]
    except KeyError:
        raise ValueError("API_KEY_2 belum terbaca. Silakan RESTART server Streamlit Anda (tekan Ctrl+C di terminal, lalu jalankan ulang 'streamlit run main.py').")
    
    if not api_key_2:
        raise ValueError("API_KEY_2 kosong di secrets.toml")
    return Groq(api_key=api_key_2)

supabase: Client = init_supabase()
gemini_model = init_gemini()
groq_client = init_groq()

# ==========================================
# 2. PROMPT SISTEM (Logika Matematika & Jenis Reksa Dana)
# ==========================================
PROMPT_SISTEM = """
Anda adalah analis data finansial tingkat lanjut. Ekstrak data Fund Fact Sheet ke dalam format JSON.

ATURAN NORMALISASI KOMPOSISI (SANGAT KRITIS):
1. Kategori Baku (Wajib 6): "Pasar Uang", "Obligasi Negara", "Obligasi Korporat", "Kas", "Saham", dan "Lainnya". Jika ada aset yang tidak cocok di 5 awal, masukkan ke "Lainnya".
2. TOTAL PORSI HARUS ~100%. Jangan menjumlahkan porsi aset utama dengan porsi alokasi sektor.
3. MATEMATIKA SUB-SEKTOR (KASUS EBU/OBLIGASI): 
   Jika dokumen mencantumkan total "Efek Bersifat Utang" (EBU) sebesar misal 54.89%, lalu di bagian lain (Alokasi Sektor) memecahnya menjadi Pemerintah 47.14% dan Perusahaan 52.86%, Anda WAJIB mengalikan keduanya agar total komposisi tidak melebihi 100%.
   Rumus: Porsi Total EBU * Porsi Sektor.
   Contoh Eksekusi:
   - Obligasi Negara = 54.89% * 47.14% = 25.87%
   - Obligasi Korporat = 54.89% * 52.86% = 29.02%
   Hitung dan masukkan hasil kalinya. JANGAN memasukkan angka 47.14% mentah-mentah.
4. Isi dengan "0.00%" jika kategori tidak ada.

STRUKTUR JSON WAJIB:
{
  "manajer_investasi": "Pilih 1 dari 11 daftar baku. WAJIB TERISI.",
  "nama_reksa_dana": "Nama lengkap produk",
  "jenis_reksa_dana": "Ekstrak kategori produk (Pasar Uang, Saham, Campuran, Pendapatan Tetap, atau Indeks)",
  "periode": "YYYY-MM-DD",
  "aum": "TULISKAN TEKS MENTAH DARI DOKUMEN, contoh: '32.83 Miliar' atau '1.2 Triliun'. Jangan ubah ke angka nol.",
  "nab_per_unit": 000.00,
  "komposisi": {
    "Pasar Uang": "0.00%",
    "Obligasi Negara": "0.00%",
    "Obligasi Korporat": "0.00%",
    "Kas": "0.00%",
    "Saham": "0.00%",
    "Lainnya": "0.00%"
  },
  "top_holdings": [
    {"instrumen": "Nama Aset", "porsi": "0.00%"}
  ]
}

INSTRUKSI UMUM:
1. nab_per_unit harus angka mentah (contoh: 1864.40).
2. EKSTRAKSI TOP HOLDINGS (SANGAT KETAT): 
   - Cari TEPAT 10 instrumen.
   - AMBIL HANYA KODE EFEK ATAU NAMA SPESIFIK EMITEN. 
   - HAPUS SEMUA KATA KELAS ASET yang menempel pada nama efek (Hapus kata seperti: "Pasar Uang", "Obligasi", "Saham", "Deposito", "TD", "Efek", "Pemerintah"). 
   - Contoh BENAR: "BANK BSI", "FR0056", "BBCA". 
   - Contoh SALAH: "Pasar Uang BANK BSI", "Obligasi FR0056", "Saham BBCA".
3. DETEKSI JENIS REKSA DANA: Cari indikator teks seperti "Jenis Reksa Dana" atau "Investment Type".
4. DETEKSI MANAJER INVESTASI: Pindai logo (header), disclaimer (footer), atau deduksi dari nama produk. Hasil harus salah satu dari: Batavia, Ashmore, BRI, BNP Paribas, Eastspring, Allianz, Maybank Asset Management, Trimegah, Schroder, Mandiri, Manulife.
"""

# ==========================================
# FUNGSI PEMBANTU FORMAT & PARSING ANGKA
# ==========================================
def format_angka(angka):
    try:
        return f"{float(angka):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "0,00"

def parse_angka_ui(teks):
    try:
        return float(str(teks).replace(".", "").replace(",", "."))
    except ValueError:
        return 0.0

def konversi_aum_llm(teks_aum):
    if isinstance(teks_aum, (int, float)):
        return float(teks_aum)
    
    teks = str(teks_aum).lower().replace(',', '.')
    pengali = 1
    
    # Menambahkan billion, million, dan trillion ke dalam deteksi
    if 'triliun' in teks or 'trillion' in teks or 't' in teks.split():
        pengali = 1_000_000_000_000
    elif 'miliar' in teks or 'milyar' in teks or 'billion' in teks or 'b' in teks.split() or 'bio' in teks:
        pengali = 1_000_000_000
    elif 'juta' in teks or 'million' in teks or 'm' in teks.split() or 'mio' in teks:
        pengali = 1_000_000
        
    match = re.search(r'([\d\.]+)', teks)
    if match:
        angka_str = match.group(1)
        if angka_str.count('.') > 1:
            angka_str = angka_str.replace('.', '', angka_str.count('.') - 1)
        try:
            return float(angka_str) * pengali
        except:
            return 0.0
    return 0.0
    
# ==========================================
# ANTARMUKA UTAMA
# ==========================================
st.title("Financial Data Intelligence")
tab_ekstraksi, tab_dasbor = st.tabs(["Ekstraksi Data", "Dasbor Analisis"])

with tab_ekstraksi:
    st.write("Arsitektur Hybrid: Gemini Vision (Primary) -> Groq LLM (Fallback)")
    uploaded_files = st.file_uploader("Pilih file PDF (Batch)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Proses Batch", type="primary"):
            with st.spinner("Menjalankan pipeline ekstraksi..."):
                extracted_list = []
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)

                for index, file in enumerate(uploaded_files):
                    pdf_bytes = file.getvalue()
                    status_placeholder = st.empty()
                    status_placeholder.info(f"[{index+1}/{total_files}] Mengirim {file.name} ke Gemini...")
                    
                    try:
                        content = [{"mime_type": "application/pdf", "data": pdf_bytes}, PROMPT_SISTEM]
                        response = gemini_model.generate_content(content, generation_config={"response_mime_type": "application/json"})
                        
                        data_json = json.loads(response.text)
                        data_json['filename'] = file.name
                        data_json['engine_used'] = "Gemini"
                        extracted_list.append(data_json)
                        status_placeholder.success(f"✓ {file.name} diproses oleh Gemini.")
                        time.sleep(3) 

                    except Exception as e_gemini:
                        if "429" in str(e_gemini) or "quota" in str(e_gemini).lower():
                            status_placeholder.warning(f"⚠ Gemini Limit tercapai. Mencoba Gemini (API Key 2)...")
                            try:
                                gemini_model_2 = init_gemini_2()
                                response_2 = gemini_model_2.generate_content(content, generation_config={"response_mime_type": "application/json"})
                                
                                data_json = json.loads(response_2.text)
                                data_json['filename'] = file.name
                                data_json['engine_used'] = "Gemini (Key 2)"
                                extracted_list.append(data_json)
                                status_placeholder.success(f"✓ {file.name} diproses oleh Gemini (Key 2).")
                                time.sleep(3)
                            except Exception as e_gemini_2:
                                if "429" in str(e_gemini_2) or "quota" in str(e_gemini_2).lower() or "rate_limit" in str(e_gemini_2).lower():
                                    status_placeholder.warning(f"⚠ Gemini (Key 2) Limit tercapai. Mengalihkan {file.name} ke Groq...")
                                    try:
                                        raw_text = ""
                                        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                                            for page in pdf.pages:
                                                ext_text = page.extract_text(layout=True)
                                                if ext_text: raw_text += ext_text + "\n"
                                        
                                        chat_completion = groq_client.chat.completions.create(
                                            messages=[{"role": "system", "content": PROMPT_SISTEM}, {"role": "user", "content": f"Teks Mentah PDF:\n\n{raw_text}"}],
                                            model="llama-3.3-70b-versatile",
                                            response_format={"type": "json_object"},
                                        )
                                        data_json = json.loads(chat_completion.choices[0].message.content)
                                        data_json['filename'] = file.name
                                        data_json['engine_used'] = "Groq"
                                        extracted_list.append(data_json)
                                        status_placeholder.success(f"✓ {file.name} diselamatkan oleh Groq.")
                                        time.sleep(1)
                                    except Exception as e_groq:
                                        if "429" in str(e_groq) or "quota" in str(e_groq).lower() or "rate_limit" in str(e_groq).lower():
                                            status_placeholder.warning(f"⚠ Groq Limit tercapai. Mengalihkan {file.name} ke Groq (API Key 2)...")
                                            try:
                                                groq_client_2 = init_groq_fallback()
                                                chat_completion_2 = groq_client_2.chat.completions.create(
                                                    messages=[{"role": "system", "content": PROMPT_SISTEM}, {"role": "user", "content": f"Teks Mentah PDF:\n\n{raw_text}"}],
                                                    model="llama-3.3-70b-versatile",
                                                    response_format={"type": "json_object"},
                                                )
                                                data_json = json.loads(chat_completion_2.choices[0].message.content)
                                                data_json['filename'] = file.name
                                                data_json['engine_used'] = "Groq (Key 2)"
                                                extracted_list.append(data_json)
                                                status_placeholder.success(f"✓ {file.name} diselamatkan oleh Groq (Key 2).")
                                                time.sleep(1)
                                            except Exception as e_groq_2:
                                                status_placeholder.error(f"❌ {file.name} gagal di semua engine (termasuk API Key 2). Error: {e_groq_2}")
                                        else:
                                            status_placeholder.error(f"❌ {file.name} gagal di kedua engine. Groq: {e_groq}")
                                else:
                                    status_placeholder.error(f"❌ {file.name} gagal di Gemini (Key 2): {e_gemini_2}")
                        else:
                            status_placeholder.error(f"❌ {file.name} gagal di Gemini: {e_gemini}")
                    
                    progress_bar.progress((index + 1) / total_files)
                
                if extracted_list:
                    st.session_state['extracted_data_list'] = extracted_list
                    st.success(f"Selesai. {len(extracted_list)} dokumen terekstrak.")

    # REVIEW & SIMPAN DATA
    if 'extracted_data_list' in st.session_state:
        st.divider()
        st.subheader("Verifikasi Kualitas Data")
        
        for i, data in enumerate(st.session_state['extracted_data_list']):
            engine_label = "🟢 Gemini" if data.get('engine_used') == "Gemini" else "🟠 Groq"
            file_key = data.get('filename', f'doc_{i}') 
            
            with st.expander(f"{engine_label} | 📄 {data.get('filename', f'Dokumen {i+1}')}", expanded=True):
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    mi_name = st.text_input("Manajer Investasi", data.get("manajer_investasi", ""), key=f"mi_{file_key}")
                    fund_name = st.text_input("Nama Produk", data.get("nama_reksa_dana", ""), key=f"fund_{file_key}")
                    
                    try:
                        parsed_date = datetime.strptime(data.get("periode", ""), "%Y-%m-%d").date()
                    except ValueError:
                        parsed_date = date.today()
                    periode = st.date_input("Periode", parsed_date, key=f"date_{file_key}")
                    
                with c2:
                    aum_mentah_ai = data.get("aum", 0)
                    aum_val = konversi_aum_llm(aum_mentah_ai)
                    
                    aum_str = st.text_input("Total AUM", value=format_angka(aum_val), key=f"aum_{file_key}")
                    aum = parse_angka_ui(aum_str)
                    
                with c3:
                    nav_val = float(data.get("nab_per_unit", 0))
                    nav_str = st.text_input("NAV per Unit", value=format_angka(nav_val), key=f"nav_{file_key}")
                    nav = parse_angka_ui(nav_str)
                    
                    fund_type = st.text_input("Jenis Reksa Dana", data.get("jenis_reksa_dana", ""), key=f"type_{file_key}")
                
                with c2:
                    tu_val = aum / nav if nav != 0 else 0.0
                    # PERBAIKAN: Gunakan key dinamis agar Streamlit dipaksa merender ulang nilai terbaru
                    st.text_input("Total Unit", value=format_angka(tu_val), key=f"tu_{file_key}_{tu_val}", disabled=True)

                st.write("#### Struktur Aset & Holdings")
                col_komp, col_hold = st.columns(2)
                with col_komp:
                    df_komp = pd.DataFrame(data.get("komposisi", {}).items(), columns=["Kategori Baku", "Porsi"])
                    edited_komp = st.data_editor(df_komp, use_container_width=True, hide_index=True, key=f"komp_{file_key}")
                with col_hold:
                    df_hold = pd.DataFrame(data.get("top_holdings", []))
                    edited_hold = st.data_editor(df_hold, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"hold_{file_key}")

                if st.button("Simpan ke Database", type="primary", key=f"btn_{file_key}"):
                    try:
                        res_mi = supabase.table("manajer_investasi").select("id").eq("nama", mi_name).execute()
                        mi_id = res_mi.data[0]["id"] if res_mi.data else supabase.table("manajer_investasi").insert({"nama": mi_name}).execute().data[0]["id"]

                        res_p = supabase.table("produk_reksadana").select("id").eq("nama_produk", fund_name).execute()
                        produk_id = res_p.data[0]["id"] if res_p.data else supabase.table("produk_reksadana").insert({"mi_id": mi_id, "nama_produk": fund_name, "kategori": fund_type}).execute().data[0]["id"]

                        formatted_date = periode.strftime("%Y-%m-%d")
                        check_exist = supabase.table("metrik_bulanan").select("id").eq("produk_id", produk_id).eq("periode", formatted_date).execute()

                        if check_exist.data:
                            st.warning(f"Data untuk '{fund_name}' periode {formatted_date} sudah ada di database. Dibatalkan.")
                        else:
                            payload_metrik = {
                                "produk_id": produk_id, "periode": formatted_date,
                                "aum": aum, "nab_per_unit": nav,
                                "komposisi": edited_komp.to_dict(orient='records'),
                                "top_holdings": edited_hold.to_dict(orient='records')
                            }
                            supabase.table("metrik_bulanan").insert(payload_metrik).execute()
                            st.success(f"Data baru {fund_name} ditambahkan.")
                    except Exception as e:
                        st.error(f"Database Error: {e}")

# ==========================================
# 4. TAB 2: DASBOR ANALISIS
# ==========================================
with tab_dasbor:
    st.subheader("Analisis Kinerja & Portofolio Historis")
    
    try:
        daftar_produk = supabase.table("produk_reksadana").select("id, nama_produk").execute().data
    except Exception as e:
        st.error(f"Gagal mengambil data produk: {e}")
        daftar_produk = []

    if daftar_produk:
        opsi_produk = {p["nama_produk"]: p["id"] for p in daftar_produk}
        produk_pilihan = st.selectbox("Pilih Reksa Dana", options=list(opsi_produk.keys()))
        
        # PERBAIKAN: Menarik seluruh kolom metrik, bukan hanya komposisi
        data_metrik = supabase.table("metrik_bulanan").select("periode, komposisi, aum, nab_per_unit, top_holdings").eq("produk_id", opsi_produk[produk_pilihan]).order("periode").execute().data

        if data_metrik:
            # Kalkulasi basis dataframe metrik mentah
            df_raw = pd.DataFrame(data_metrik)
            df_raw['Total Unit'] = df_raw.apply(lambda x: x['aum'] / x['nab_per_unit'] if x['nab_per_unit'] > 0 else 0, axis=1)

            # Membagi Dasbor menjadi 3 sub-tab agar rapi
            subtab_alokasi, subtab_kinerja, subtab_holdings = st.tabs(["Alokasi Aset", "Tren AUM & NAV", "Top 10 Holdings"])

            # ------------------------------------
            # SUB-TAB 1: ALOKASI ASET
            # ------------------------------------
            with subtab_alokasi:
                baris_data = []
                for row in data_metrik:
                    for item in row.get("komposisi", []):
                        porsi_str = str(item.get("Porsi", "0")).replace("%", "").strip()
                        try:
                            porsi_float = float(porsi_str)
                        except ValueError:
                            porsi_float = 0.0
                            
                        baris_data.append({
                            "Periode": row["periode"],
                            "Kategori Aset": item.get("Kategori Baku", ""),
                            "Persentase (%)": porsi_float
                        })

                if baris_data:
                    df_komposisi = pd.DataFrame(baris_data).sort_values("Periode")
                    fig = px.bar(
                        df_komposisi, x="Periode", y="Persentase (%)", color="Kategori Aset",
                        title=f"Evolusi Alokasi Aset: {produk_pilihan}", text_auto='.2f',
                        category_orders={"Kategori Aset": ["Saham", "Obligasi Korporat", "Obligasi Negara", "Pasar Uang", "Lainnya", "Kas"]}
                    )
                    fig.update_layout(barmode='stack', hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------
            # SUB-TAB 2: TREN KINERJA (AUM, NAV, UNIT)
            # ------------------------------------
            with subtab_kinerja:
                # Membuat subplot dengan 2 y-axis
                fig_gabungan = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Tambahkan AUM sebagai Bar Chart (y-axis utama)
                fig_gabungan.add_trace(
                    go.Bar(x=df_raw["periode"], y=df_raw["aum"], name="AUM", marker_color="royalblue", opacity=0.7),
                    secondary_y=False,
                )
                
                # Tambahkan NAV sebagai Line Chart (y-axis sekunder)
                fig_gabungan.add_trace(
                    go.Scatter(x=df_raw["periode"], y=df_raw["nab_per_unit"], name="NAV per Unit", mode="lines+markers", marker_color="orange"),
                    secondary_y=True,
                )
                
                # Konfigurasi layout
                fig_gabungan.update_layout(
                    title_text="Pertumbuhan AUM dan NAV per Unit",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Set judul axis
                fig_gabungan.update_yaxes(title_text="AUM", secondary_y=False)
                fig_gabungan.update_yaxes(title_text="NAV per Unit", secondary_y=True)
                
                st.plotly_chart(fig_gabungan, use_container_width=True)

                fig_unit = px.line(df_raw, x="periode", y="Total Unit", title="Total Unit Beredar", markers=True)
                st.plotly_chart(fig_unit, use_container_width=True)

            # ------------------------------------
            # SUB-TAB 3: EVOLUSI TOP HOLDINGS
            # ------------------------------------
            with subtab_holdings:
                st.markdown("Tabel matriks ini menunjukkan pergeseran bobot instrumen dari bulan ke bulan.")
                
                baris_holdings = []
                for row in data_metrik:
                    for item in row.get("top_holdings", []):
                        # Memaksa string menjadi kapital (UPPERCASE) untuk mengurangi risiko duplikasi akibat huruf kecil/besar
                        instrumen_nama = str(item.get("instrumen", "")).strip().upper()
                        baris_holdings.append({
                            "Periode": row["periode"],
                            "Instrumen": instrumen_nama,
                            "Porsi": item.get("porsi", "0.00%")
                        })

                if baris_holdings:
                    df_holdings = pd.DataFrame(baris_holdings)
                    
                    # Membuat Pivot Table: Baris = Instrumen, Kolom = Periode
                    df_pivot_holdings = df_holdings.pivot(index="Instrumen", columns="Periode", values="Porsi").fillna("-")
                    
                    st.dataframe(df_pivot_holdings, use_container_width=True)
                else:
                    st.info("Data Top Holdings belum tersedia untuk ditarik.")

        else:
            st.info("Belum ada data metrik historis untuk produk ini.")