# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import io
from datetime import datetime, timedelta
import plotly.express as px

# ---------- CONFIG ----------
st.set_page_config(page_title="PredictX Pro Ultra V3", layout="wide", page_icon="⚽")
DATA_FILE = "predictions.csv"
API_KEY = "YOUR_API_KEY_HERE"   # <<<--- Replace with your API-FOOTBALL key (keep quotes)
API_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# ---------- DATA HELPERS ----------
def ensure_data_file():
    """Create the CSV with correct columns if it doesn't exist."""
    if not os.path.exists(DATA_FILE):
        df0 = pd.DataFrame(columns=[
            "id", "date", "team_a", "team_b", "predicted_winner",
            "prediction_method", "confidence", "actual_result",
            "outcome", "notes", "created_at", "updated_at"
        ])
        df0.to_csv(DATA_FILE, index=False)

def load_data():
    """Return a DataFrame (never None)."""
    ensure_data_file()
    df = pd.read_csv(DATA_FILE, dtype=str)
    expected = ["id","date","team_a","team_b","predicted_winner","prediction_method",
                "confidence","actual_result","outcome","notes","created_at","updated_at"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df

def save_data(df):
    """Persist DataFrame to CSV safely."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("save_data expects a pandas DataFrame, got: %s" % type(df))
    df.to_csv(DATA_FILE, index=False)

# ---------- API HELPERS ----------
def api_get(endpoint, params=None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", headers=HEADERS, params=(params or {}), timeout=12)
        # If API returns non-JSON or error, return an object indicating error
        try:
            data = resp.json()
        except:
            return {"error": f"Non-JSON response, status {resp.status_code}"}
        if resp.status_code != 200:
            return {"error": f"Status {resp.status_code}", "detail": data}
        return data
    except Exception as e:
        return {"error": str(e)}

def find_team_id(team_name):
    """Find best matching team id and canonical name, or return (None, None)."""
    data = api_get("/teams", params={"search": team_name})
    if not data or "error" in data:
        return None, None
    resp = data.get("response", [])
    if not resp:
        return None, None
    team_obj = resp[0].get("team", {})
    return team_obj.get("id"), team_obj.get("name")

def get_recent_matches(team_id, last_n=10):
    data = api_get("/fixtures", params={"team": team_id, "last": last_n})
    if not data or "error" in data:
        return []
    return data.get("response", [])

def def get_team_id(team_name):
    # Try to find the team globally
    url = f"https://v3.football.api-sports.io/teams?search={team_name}"
    headers = {"x-apisports-key": API_KEY}
    response = requests.get(url, headers=headers).json()

    if response["response"]:
        return response["response"][0]["team"]["id"]
    else:
        return None


team_a_id = get_team_id(team_a)
team_b_id = get_team_id(team_b)

if not team_a_id or not team_b_id:
    return {"error": f"Could not find one or both teams via API. Try clearer names or check spelling."} last_n=10):
    data = api_get("/fixtures", params={"h2h": f"{team_a_id}-{team_b_id}", "last": last_n})
    if not data or "error" in data:
        return []
    return data.get("response", [])

# ---------- PREDICTION LOGIC ----------
def compute_team_score(team_id):
    """Compute simple heuristic score using recent matches (goals & form)."""
    stats = {"goals_for": 0, "goals_against": 0, "wins": 0, "draws": 0, "losses": 0, "matches": 0}
    matches = get_recent_matches(team_id, last_n=10)
    if not matches:
        return stats, 0.0
    for m in matches:
        goals = m.get("goals", {})
        teams = m.get("teams", {})
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")
        home_goals = goals.get("home")
        away_goals = goals.get("away")
        # skip unfinished fixtures
        if home_goals is None or away_goals is None:
            continue
        if team_id == home_id:
            gf, ga = home_goals, away_goals
            if home_goals > away_goals:
                stats["wins"] += 1
            elif home_goals == away_goals:
                stats["draws"] += 1
            else:
                stats["losses"] += 1
        else:
            gf, ga = away_goals, home_goals
            if away_goals > home_goals:
                stats["wins"] += 1
            elif away_goals == home_goals:
                stats["draws"] += 1
            else:
                stats["losses"] += 1
        stats["matches"] += 1
        stats["goals_for"] += (gf or 0)
        stats["goals_against"] += (ga or 0)
    if stats["matches"] > 0:
        form_score = (stats["wins"]*3 + stats["draws"]) / (stats["matches"]*3)
    else:
        form_score = 0
    gf_per = stats["goals_for"] / stats["matches"] if stats["matches"]>0 else 0
    score = 0.5 * gf_per + 0.5 * form_score * 3
    return stats, score

def compute_prediction_fixed(team_a, team_b):
    """
    Return stored fixed prediction if exists, else compute using API and save it.
    Returns dict with saved fields or {'error':...}
    """
    df = load_data()
    a = team_a.strip()
    b = team_b.strip()

    # Try finding stored exact match (either order)
    match_row = df[
        ((df['team_a'].str.lower() == a.lower()) & (df['team_b'].str.lower() == b.lower())) |
        ((df['team_a'].str.lower() == b.lower()) & (df['team_b'].str.lower() == a.lower()))
    ]
    if not match_row.empty:
        # return first stored record
        row = match_row.iloc[0].to_dict()
        return row

    # No stored prediction - compute now
    ta_id, ta_name = find_team_id(a)
    tb_id, tb_name = find_team_id(b)
    if not ta_id or not tb_id:
        return {"error": "Could not find one or both teams via API."}

    stats_a, score_a = compute_team_score(ta_id)
    stats_b, score_b = compute_team_score(tb_id)

    h2h = get_head_to_head(ta_id, tb_id, last_n=10)
    a_wins = b_wins = draws = 0
    if h2h:
        for m in h2h:
            goals = m.get("goals", {})
            teams = m.get("teams", {})
            if goals.get("home") is None or goals.get("away") is None:
                continue
            home_id = teams.get("home", {}).get("id")
            away_id = teams.get("away", {}).get("id")
            if (home_id == ta_id and goals.get("home") > goals.get("away")) or (away_id == ta_id and goals.get("away") > goals.get("home")):
                a_wins += 1
            elif (home_id == tb_id and goals.get("home") > goals.get("away")) or (away_id == tb_id and goals.get("away") > goals.get("home")):
                b_wins += 1
            else:
                draws += 1
    h2h_adv = 0
    denom = (a_wins + b_wins + draws)
    if denom > 0:
        h2h_adv = (a_wins - b_wins) / denom

    combined_a = score_a + 0.5 * h2h_adv
    combined_b = score_b - 0.5 * h2h_adv

    diff = combined_a - combined_b
    confidence = float(min(0.99, max(0.01, 0.5 + diff/4)))
    if combined_a > combined_b:
        predicted = a
    elif combined_b > combined_a:
        predicted = b
    else:
        predicted = "Draw"

    # Prepare new row
    df = load_data()
    try:
        existing_ids = pd.to_numeric(df['id'], errors='coerce')
        max_id = int(existing_ids.max()) if not existing_ids.empty and not pd.isna(existing_ids.max()) else 0
    except Exception:
        max_id = len(df)
    new_id = int(max_id) + 1
    now = datetime.utcnow().isoformat()
    new_row = {
        "id": new_id,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "team_a": a,
        "team_b": b,
        "predicted_winner": predicted,
        "prediction_method": "stats_h2h",
        "confidence": round(confidence, 3),
        "actual_result": "",
        "outcome": "",
        "notes": f"score_a:{round(combined_a,3)} score_b:{round(combined_b,3)}",
        "created_at": now,
        "updated_at": now
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)
    return new_row

# ---------- UI ----------
st.markdown("<style> .card{background: rgba(10,14,23,0.75); padding:12px; border-radius:10px} </style>", unsafe_allow_html=True)
st.title("⚽ PredictX Pro Ultra V3")
st.caption("AI-Powered Football Prediction Analyzer — Predictive Hub & Review System")

# Sidebar/Top navigation using tabs
menu = st.tabs(["⚽ Predictive Hub", "📘 Reviews Book", "📊 Performance Chart"])

# ---------------- Predictive Hub ----------------
with menu[0]:
    st.header("⚽ Predictive Hub")
    st.markdown("Choose a live match or enter teams manually. Predictions are stored and fixed after creation.")

    # Live fixture picker
    team1 = ""
    team2 = ""
    try:
        today = datetime.utcnow().date()
        fixture_resp = api_get("/fixtures", params={"date": today.isoformat()})
        if fixture_resp and "error" not in fixture_resp and fixture_resp.get("results", 0) > 0:
            matches = []
            for m in fixture_resp.get("response", []):
                home = m.get("teams", {}).get("home", {}).get("name")
                away = m.get("teams", {}).get("away", {}).get("name")
                league = m.get("league", {}).get("name")
                matches.append({"label": f"{home} vs {away} — {league}", "home": home, "away": away})
            labels = [x["label"] for x in matches]
            sel = st.selectbox("Select today's match (or leave blank to enter manually)", [""] + labels)
            if sel:
                chosen = next((x for x in matches if x["label"] == sel), None)
                if chosen:
                    team1 = chosen["home"]
                    team2 = chosen["away"]
                    st.info(f"Selected: {team1} vs {team2}")
        else:
            st.info("No fixtures found for today (or API returned none). You may enter teams manually.")
    except Exception as e:
        st.warning("Live fixtures unavailable: " + str(e))
        st.info("You may enter teams manually.")

    # manual fallback inputs if not selected
    if not team1:
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.text_input("Home team", value="")
        with col2:
            team2 = st.text_input("Away team", value="")

    if st.button("Get Prediction"):
        if not team1 or not team2:
            st.warning("Enter both teams.")
        else:
            with st.spinner("Computing prediction..."):
                res = compute_prediction_fixed(team1, team2)
                if isinstance(res, dict) and res.get("error"):
                    st.error("Error: " + str(res.get("error")))
                else:
                    st.success(f"Predicted Winner: **{res.get('predicted_winner')}** (confidence: {res.get('confidence')})")
                    st.markdown("**Prediction details:**")
                    st.write("Method:", res.get("prediction_method"))
                    st.write("Notes:", res.get("notes"))
                    st.write("Saved prediction ID:", res.get("id"))

# ---------------- Reviews Book ----------------
with menu[1]:
    st.header("📘 Reviews Book")
    st.markdown("All saved predictions. Update with actual results and mark Correct or Wrong.")

    df = load_data()
    if df.empty:
        st.info("No predictions yet.")
    else:
        df_display = df.copy()
        display_cols = ["id","date","team_a","team_b","predicted_winner","confidence","actual_result","outcome","notes","created_at"]
        st.dataframe(df_display[display_cols].sort_values("created_at", ascending=False))

        st.markdown("### Update a prediction result")
        sid = st.text_input("Enter prediction ID to update (see table above)", value="")
        if sid:
            try:
                sid_int = int(sid)
                idx = df.index[df['id'].astype(int) == sid_int]
                if idx.empty:
                    st.error("ID not found.")
                else:
                    i = idx[0]
                    st.write("Selected:", df.at[i, "team_a"], "vs", df.at[i, "team_b"], " | Predicted:", df.at[i, "predicted_winner"])
                    actual = st.text_input("Enter actual result (e.g. 2-1 or Draw)", value=df.at[i, "actual_result"])
                    outcome = st.selectbox("Outcome", ["", "Correct", "Wrong", "Draw"], index=0)
                    note = st.text_area("Notes (optional)", value=df.at[i, "notes"])
                    if st.button("Save result"):
                        df.at[i, 'actual_result'] = actual
                        df.at[i, 'outcome'] = outcome
                        df.at[i, 'notes'] = note
                        df.at[i, 'updated_at'] = datetime.utcnow().isoformat()
                        save_data(df)
                        st.success("Prediction updated.")
                        st.experimental_rerun()
            except ValueError:
                st.error("Please enter a numeric ID.")

        # Download CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("Download predictions CSV", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")

# ---------------- Performance Chart ----------------
with menu[2]:
    st.header("📊 Performance Chart")
    st.markdown("See how your predictions performed over time.")

    df = load_data()
    if df.empty or df['outcome'].isin(["Correct","Wrong"]).sum() == 0:
        st.info("No completed results to chart yet. Mark outcomes in Reviews Book.")
    else:
        chart_df = df[df['outcome'].isin(["Correct","Wrong"])].copy()
        chart_df['created_at'] = pd.to_datetime(chart_df['created_at'])
        chart_df = chart_df.sort_values('created_at')
        chart_df['is_correct'] = (chart_df['outcome'] == "Correct").astype(int)
        # daily accuracy
        summary = chart_df.groupby(pd.Grouper(key='created_at', freq='D')).agg({'is_correct':'mean','id':'count'}).reset_index()
        summary = summary.rename(columns={'is_correct':'accuracy','id':'matches'})
        if summary.empty:
            st.info("Not enough data yet to show chart.")
        else:
            fig = px.line(summary, x='created_at', y='accuracy', title='Daily Accuracy (fraction correct)')
            fig.update_yaxes(tickformat=".0%", range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

            # simple overall metrics
            total = len(chart_df)
            correct = int(chart_df['is_correct'].sum())
            accuracy = correct / total * 100
            st.metric("Total Marked Matches", total)
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")
