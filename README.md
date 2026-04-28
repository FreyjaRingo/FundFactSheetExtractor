# Fund Fact Sheet Extractor (Financial Data Intelligence)

A robust Streamlit application designed to automatically extract, validate, and visualize mutual fund data from Fund Fact Sheet PDFs. The system utilizes a hybrid AI architecture (Google Gemini Vision as primary, with Groq Llama 3 as fallback) to ensure high-quality data extraction and integrates seamlessly with Supabase for data storage and analysis.

## Features

- **Batch PDF Processing**: Upload multiple Fund Fact Sheet PDFs and extract key metrics (AUM, NAV, Composition, Top Holdings) simultaneously.
- **Hybrid AI Pipeline**: Intelligent extraction pipeline using Gemini-2.5-flash and Groq Llama-3.3-70b to handle complex layouts and rate limits.
- **Data Validation & Normalization**: Built-in logic to validate asset allocations (ensuring 100% total) and parse complex financial numbers securely.
- **Automated Database Sync**: Stores all extracted metrics in a relational Supabase database.
- **Interactive Analytics Dashboard**: Beautiful Plotly visualizations to track Asset Allocation evolution, AUM & NAV growth (Dual-axis charts), and Top 10 Holdings shifts over time.

## Requirements

The project relies on Python 3.9+ and the dependencies listed in `requirements.txt`. Key libraries include:
- `streamlit`
- `pandas`
- `plotly`
- `pdfplumber`
- `groq`
- `supabase`
- `google-generativeai`

## Deployment to Streamlit Community Cloud

Follow these steps to deploy the application to Streamlit Community Cloud:

1. **Push your code to GitHub**: Make sure `main.py` and `requirements.txt` are in the root directory of your repository.
2. **Create a new Streamlit app**: 
   - Go to [share.streamlit.io](https://share.streamlit.io/) and click "New app".
   - Select your GitHub repository, branch, and set the Main file path to `main.py` (or the correct path in your repo).
3. **Configure Secrets**: 
   - Before clicking "Deploy", click on **"Advanced settings..."** (or go to App settings -> Secrets after deployment).
   - Add your API keys and database credentials in TOML format:

```toml
[supabase]
URL = "your_supabase_project_url"
KEY = "your_supabase_anon_key"

[gemini]
API_KEY = "your_primary_gemini_api_key"
API_KEY_2 = "your_fallback_gemini_api_key"

[groq]
API_KEY = "your_primary_groq_api_key"
API_KEY_2 = "your_fallback_groq_api_key"
```

4. **Deploy**: Click "Deploy" and Streamlit will automatically install the packages from `requirements.txt` and launch your app.
