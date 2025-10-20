# ⚽ PredictX Pro Ultra V2 — Football Prediction Dashboard

An interactive web dashboard that lets you log match predictions, check actual results, and visualize accuracy trends.

## 🚀 Deployment (Streamlit Cloud)
1. Create a new GitHub repository (e.g., `predictx-pro-ultra-v2`) and upload the files (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **New app** → Select your repository → File path: `app.py` → Branch: `main` → Deploy.

Your app will be live at a Streamlit URL like:
```
https://<your-username>.streamlit.app
```

## 🧰 Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Features
- Predictions table with confidence levels
- Real result input & automatic comparison
- Dashboard with accuracy %, pie charts, and trends
- Review log of matches
- Download combined reports (CSV/Excel)
