
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
API_KEY = "2debb76dcc5bbe808e64d70de9b17abf"
API_BASE = "https://v3.football.api-sports.io"

HEADERS = {"x-apisports-key": API_KEY}

# ---------- UTILITIES ----------
def ensure_data_file():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=[
            "id", "date", "team_a", "team_b", "predicted_winner",
            "prediction_method", "confidence", "actual_result",
            "outcome", "notes", "created_at", "updated_at"
        ])
        df.to_csv(DATA_FILE, index=False)

def load_data():
    ensure_data_file()
    return pd.read_csv(DATA_FILE, dtype=str)

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def normalize_team_name(name):
    return name.strip()

# API helpers with graceful handling
def api_get(endpoint, params=None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", headers=HEADERS, params=(params or {}), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"API status {resp.status_code}", "detail": resp.text}
    except Exception as e:
        return {"error": str(e)}

def find_team_id(team_name):
    try:
        data = api_get("/teams", params={"search": team_name})
        if data is None or "error" in data:
            return None
        resp = data.get("response", [])
        if not resp:
            return None
        # pick best match (first)
        return resp[0]["team"]["id"], resp[0]["team"]["name"]
    except:
        return None

def get_recent_matches(team_id, last_n=10):
    # Try last season and recent fixtures: use last N fixtures endpoint (fixtures)
    params = {"team": team_id, "last": last_n}
    data = api_get("/fixtures", params=params)
    if data is None or "error" in data:
        return None
    return data.get("response", [])

def get_head_to_head(team_a_id, team_b_id, last_n=10):
    params = {"h2h": f"{team_a_id}-{team_b_id}", "last": last_n}
    data = api_get("/fixtures", params=params)
    if data is None or "error" in data:
        return None
    return data.get("response", [])

# Simple scoring function using goals, recent form and h2h
def compute_team_score(team_id):
    # returns a dict of stats and score
    stats = {"goals_for": 0, "goals_against": 0, "wins": 0, "draws": 0, "losses": 0, "matches": 0, "form_score": 0}
    matches = get_recent_matches(team_id, last_n=10)
    if not matches:
        return stats, 0.0
    for m in matches:
        # determine if team is home or away
        fixture = m.get("fixture", {})
        teams = m.get("teams", {})
        goals = m.get("goals", {})
        home = teams.get("home", {}).get("id")
        away = teams.get("away", {}).get("id")
        home_goals = goals.get("home")
        away_goals = goals.get("away")
        # if fixture not finished, skip
        if home_goals is None or away_goals is None:
            continue
        stats["matches"] += 1
        if team_id == home:
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
        stats["goals_for"] += gf
        stats["goals_against"] += ga
    # form score: wins*3 + draws*1, normalized
    if stats["matches"] > 0:
        stats["form_score"] = (stats["wins"]*3 + stats["draws"]) / (stats["matches"]*3)
    # final heuristic score: weights to goals_for per match + form score
    gf_per_match = stats["goals_for"] / stats["matches"] if stats["matches"]>0 else 0
    score = 0.5 * gf_per_match + 0.5 * stats["form_score"] * 3
    return stats, score

def compute_prediction_fixed(team_a, team_b):
    """
    If stored prediction exists for exact (team_a vs team_b), return it.
    Otherwise compute via API stats and save it (fixed).
    """
    df = load_data()
    # canonical key (order-insensitive or enforce A vs B as given)
    # We'll store as team_a|team_b as provided (case-insensitive trimmed)
    a = normalize_team_name(team_a)
    b = normalize_team_name(team_b)
    # look for existing row with same teams (both directions)
    row = df[(df['team_a'].str.lower() == a.lower()) & (df['team_b'].str.lower() == b.lower())]
    if row.empty:
        # try reverse
        row = df[(df['team_a'].str.lower() == b.lower()) & (df['team_b'].str.lower() == a.lower())]
        if not row.empty:
            # if stored with reverse, return stored prediction (but flip perspective)
            stored = row.iloc[0].to_dict()
            return stored
    else:
        stored = row.iloc[0].to_dict()
        return stored

    # No stored record -> compute using API
    ta = find_team_id(a)
    tb = find_team_id(b)
    if not ta or not tb:
        return {"error": "Team not found"}

    team_a_id, team_a_name = ta
    team_b_id, team_b_name = tb

    # compute stats and scores
    stats_a, score_a = compute_team_score(team_a_id)
    stats_b, score_b = compute_team_score(team_b_id)

    # head-to-head advantage
    h2h = get_head_to_head(team_a_id, team_b_id, last_n=10)
    h2h_adv = 0
    if h2h:
        # count results favouring a vs b
        a_wins = 0
        b_wins = 0
        draws = 0
        for m in h2h:
            goals = m.get("goals", {})
            teams = m.get("teams", {})
            h_id = teams.get("home", {}).get("id")
            a_id = team_a_id
            # ensure finished
            if goals.get("home") is None or goals.get("away") is None:
                continue
            if (m.get('teams', {}).get('home', {}).get('id') == team_a_id and goals['home'] > goals['away']) or \
               (m.get('teams', {}).get('away', {}).get('id') == team_a_id and goals['away'] > goals['home']):
                a_wins += 1
            elif (m.get('teams', {}).get('home', {}).get('id') == team_b_id and goals['home'] > goals['away']) or \
                 (m.get('teams', {}).get('away', {}).get('id') == team_b_id and goals['away'] > goals['home']):
                b_wins += 1
            else:
                draws += 1
        if (a_wins + b_wins + draws) > 0:
            h2h_adv = (a_wins - b_wins) / max(1, (a_wins + b_wins + draws))

    # final combined score with weights
    combined_a = score_a + 0.5 * h2h_adv
    combined_b = score_b - 0.5 * h2h_adv

    # confidence metric (0-1)
    diff = combined_a - combined_b
    confidence = float(min(0.99, max(0.01, 0.5 + diff/4)))  # map diff to 0-1 roughly

    if combined_a > combined_b:
        predicted = team_a
    elif combined_b > combined_a:
        predicted = team_b
    else:
        predicted = "Draw"

    # Save to CSV as fixed prediction
    df = load_data()
    new_id = 1
    if not df.empty:
        try:
            new_id = int(df['id'].astype(int).max()) + 1
        except:
            new_id = len(df) + 1
    now = datetime.utcnow().isoformat()
    new_row = {
        "id": new_id,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "team_a": team_a,
        "team_b": team_b,
        "predicted_winner": predicted,
        "prediction_method": "stats_h2h",
        "confidence": round(confidence, 3),
        "actual_result": "",
        "outcome": "",
        "notes": f"score_a:{round(combined_a,3)} score_b:{round(combined_b,3)}",
        "created_at": now,
        "updated_at": now
 df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)
    return new_row

# ---------- UI PAGES ----------
st.markdown("<style> .card{background: rgba(10,14,23,0.75); padding:12px; border-radius:10px} </style>", unsafe_allow_html=True)
st.title("⚽ PredictX Pro Ultra V3")
st.caption("AI-Powered Football Prediction Analyzer — Predictive Hub & Review System")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Match Predictor", "Reviews Book", "Predictive Hub"])

# ---------------- Match Predictor ----------------
if page == "Match Predictor":
    st.header("Match Predictor")
    st.markdown("Enter teams and get a fixed prediction (uses live stats & head-to-head).")
    col1, col2 = st.columns([2,2])
    with col1:
        team_a = st.text_input("Team A (home)", value="")
    with col2:
        team_b = st.text_input("Team B (away)", value="")
    if st.button("Get Prediction"):
        if not team_a or not team_b:
            st.warning("Please enter both team names.")
        else:
            with st.spinner("Fetching data and computing prediction..."):
                res = compute_prediction_fixed(team_a, team_b)
                if isinstance(res, dict) and res.get("error"):
                    st.error("Error: " + str(res.get("error")))
                else:
                    st.success(f"Predicted Winner: **{res['predicted_winner']}** (confidence: {res['confidence']})")
                    # show summary stats if available
                    st.markdown("**Prediction details:**")
                    st.write("Method:", res.get("prediction_method"))
                    st.write("Notes:", res.get("notes"))
                    st.write("Saved as a fixed prediction (it will not change unless you update it in Reviews Book).")

# ---------------- Reviews Book ----------------
elif page == "Reviews Book":
    st.header("Reviews Book")
    st.markdown("View all saved predictions. Mark results when real match outcome is known.")
    df = load_data()
    if df.empty:
        st.info("No predictions yet.")
    else:
        # Show table and allow selection
        st.dataframe(df[['id','date','team_a','team_b','predicted_winner','confidence','actual_result','outcome']])
        st.markdown("### Update a prediction result")
        sid = st.text_input("Enter prediction ID to update (see table above)", value="")
        if sid:
            try:
                sid_int = int(sid)
            except:
                st.error("Enter a numeric ID.")
                sid_int = None
            if sid_int:
                row = df[df['id'].astype(int) == sid_int]
                if row.empty:
                    st.error("ID not found.")
                else:
                    selected = row.iloc[0].to_dict()
                    st.write("Selected:", selected['team_a'], "vs", selected['team_b'], "| Predicted:", selected['predicted_winner'])
                    actual = st.text_input("Enter actual result (e.g. 2-1 or Draw)", value=selected.get('actual_result',''))
                    outcome = st.selectbox("Outcome", ["", "Correct", "Wrong", "Draw"], index=0)
                    note = st.text_area("Notes (optional)", value=selected.get('notes',''))
                    if st.button("Save result"):
                        # update
                        idx = df.index[df['id'].astype(int) == sid_int][0]
                        df.at[idx, 'actual_result'] = actual
                        df.at[idx, 'outcome'] = outcome
                        df.at[idx, 'notes'] = note
                        df.at[idx, 'updated_at'] = datetime.utcnow().isoformat()
                        save_data(df)
                        st.success("Prediction updated.")
                        st.experimental_rerun()
        # Download CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("Download predictions CSV", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")
        # Show accuracy chart
        st.markdown("### Accuracy Chart")
        chart_df = df[df['outcome'].isin(["Correct","Wrong"])]
        if chart_df.empty:
            st.info("No results marked yet. Mark outcomes to see accuracy over time.")
        else:
            chart_df['created_at'] = pd.to_datetime(chart_df['created_at'])
            chart_df = chart_df.sort_values('created_at')
            chart_df['is_correct'] = chart_df['outcome'] == "Correct"
            summary = chart_df.groupby(pd.Grouper(key='created_at', freq='D')).agg({'is_correct':'mean','id':'count'}).reset_index()
            summary = summary.rename(columns={'is_correct':'accuracy','id':'matches'})
            fig = px.bar(summary, x='created_at', y='accuracy', labels={'created_at':'Date','accuracy':'Accuracy'}, title='Daily Accuracy (Correct / Total)')
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Predictive Hub (advanced analytics) ----------------
elif page == "Predictive Hub":
    st.header("Predictive Hub — in-depth team analysis")
    st.markdown("Compare historical form, goals, head-to-head, and see why the model predicted a team.")
    ta = st.text_input # --- LIVE FIXTURE PICKER ---
st.subheader("📅 Choose a Live or Upcoming Match")

headers = {"x-apisports-key": API_KEY}
base_url = "https://v3.football.api-sports.io"

try:
    import datetime as dt
    today = dt.date.today()

    response = requests.get(f"{base_url}/fixtures?date={today}", headers=headers)
    data = response.json()

    if data["results"] > 0:
        matches = []
        for match in data["response"]:
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]
            league = match["league"]["name"]
            matches.append(f"{home} vs {away} — {league}")

        selected_match = st.selectbox("Select a match", matches)
        if selected_match:
            parts = selected_match.split(" vs ")
            team1 = parts[0].strip()
            team2 = parts[1].split(" — ")[0].strip()
            st.info(f"You selected **{team1} vs {team2}** from {match['league']['name']}")
    else:
        st.warning("No live fixtures found for today. You can still enter teams manually.")
        team1 = st.text_input("Enter Home Team")
        team2 = st.text_input("Enter Away Team")

except Exception as e:
    st.error(f"Error fetching live fixtures: {e}")
    team1 = st.text_input("Enter Home Team")
    team2 = st.text_input("Enter Away Team")
    tb = st.text_input
    if st.button("Analyze teams"):
        if not ta or not tb:
            st.warning("Enter both teams.")
        else:
            with st.spinner("Fetching deep stats..."):
                ida = find_team_id(ta)
                idb = find_team_id(tb)
                if not ida or not idb:
                    st.error("Could not find one or both teams. Try alternative names.")
                else:
                    ida_id, ida_name = ida
                    idb_id, idb_name = idb
                    st.subheader(f"{ida_name} vs {idb_name}")
                    # team A stats
                    stats_a, score_a = compute_team_score(ida_id)
                    stats_b, score_b = compute_team_score(idb_id)
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown(f"### {ida_name} — Recent summary")
                        st.write(stats_a)
                        st.write("Heuristic score:", round(score_a,3))
                    with colB:
                        st.markdown(f"### {idb_name} — Recent summary")
                        st.write(stats_b)
                        st.write("Heuristic score:", round(score_b,3))
                    # head to head
                    h2h = get_head_to_head(ida_id, idb_id, last_n=10)
                    if h2h:
                        st.markdown("### Head-to-Head (last matches)")
                        hdf_rows = []
                        for m in h2h:
                            fixture = m.get('fixture',{})
                            date = fixture.get('date','')[:10]
                            teams = m.get('teams',{})
                            goals = m.get('goals',{})
                            home = teams.get('home', {}).get('name')
                            away = teams.get('away', {}).get('name')
                            score = score = f"{goals.get('home')}-{goals.get('away')}"
                            hdf_rows.append({"date":date,"home":home,"away":away,"score":score})
                        hdf = pd.DataFrame(hdf_rows)
                        st.dataframe(hdf)
                    else:
                        st.info("No head-to-head data available.")
                    # Suggestion
                    if score_a > score_b:
                        st.success(f"Model suggests: **{ta}** more likely (score {round(score_a,3)} vs {round(score_b,3)})")
                    elif score_b > score_a:
                        st.success(f"Model suggests: **{tb}** more likely (score {round(score_b,3)} vs {round(score_a,3)})")
                    else:
                        st.info("Too close to call — consider manual review.")
