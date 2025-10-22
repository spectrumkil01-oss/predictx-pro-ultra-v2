
# app.py — PredictX Pro Ultra V3 (with separate Live Fixtures page)
import streamlit as st
import pandas as pd
import requests
import os
import io
from datetime import datetime, timedelta, date
import plotly.express as px

# ---------- CONFIG ----------
st.set_page_config(page_title="PredictX Pro Ultra V3", layout="wide", page_icon="⚽")
DATA_FILE = "predictions.csv"

# <<< YOUR API-FOOTBALL KEY (keep private) >>>
API_KEY = "2debb76dcc5bbe808e64d70de9b17abf"
API_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# ---------- STYLES (Dark) ----------
st.markdown("""
    <style>
    :root{ --card-bg: rgba(10,14,23,0.85); --muted:#94a3b8; --accent:#2da8ff; }
    body { background-color: #020617; color: #dbeafe; }
    .stApp { background: linear-gradient(180deg, rgba(2,6,12,0.95), rgba(4,10,20,0.95)); }
    .card { background: var(--card-bg); padding: 14px; border-radius: 12px; box-shadow: 0 6px 30px rgba(2,6,23,0.6); }
    .title { font-size:22px; font-weight:700; color:#e6f6ff; }
    footer { color: #8298a6; text-align:center; padding:8px; font-size:12px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">⚽ PredictX Pro Ultra V3</div>', unsafe_allow_html=True)
st.caption("AI-Powered Football Prediction Analyzer — Predictive Hub & Live Fixtures")

# ---------- DATA HELPERS ----------
def ensure_data_file():
    if not os.path.exists(DATA_FILE):
        df0 = pd.DataFrame(columns=[
            "id", "date", "team_a", "team_b", "predicted_winner",
            "prediction_method", "confidence", "actual_result",
            "outcome", "notes", "created_at", "updated_at"
        ])
        df0.to_csv(DATA_FILE, index=False)

def load_data():
    ensure_data_file()
    df = pd.read_csv(DATA_FILE, dtype=str)
    expected = ["id","date","team_a","team_b","predicted_winner","prediction_method",
                "confidence","actual_result","outcome","notes","created_at","updated_at"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df

def save_data(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("save_data expects a pandas DataFrame, got: %s" % type(df))
    df.to_csv(DATA_FILE, index=False)

# ---------- API HELPERS ----------
def api_get(endpoint, params=None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", headers=HEADERS, params=(params or {}), timeout=12)
        try:
            data = resp.json()
        except Exception:
            return {"error": f"Non-JSON response, status {resp.status_code}"}
        if resp.status_code != 200:
            return {"error": f"Status {resp.status_code}", "detail": data}
        return data
    except Exception as e:
        return {"error": str(e)}

def find_team_candidates(query, limit=8):
    data = api_get("/teams", params={"search": query})
    if not data or "error" in data:
        return []
    resp = data.get("response", [])
    names = []
    for r in resp[:limit]:
        t = r.get("team", {})
        names.append(t.get("name"))
    return names

def find_best_team_id(query):
    data = api_get("/teams", params={"search": query})
    if data and "error" not in data:
        resp = data.get("response", [])
        if resp:
            team_obj = resp[0].get("team", {})
            return team_obj.get("id"), team_obj.get("name")
    return None, None

def get_recent_matches(team_id, last_n=10):
    data = api_get("/fixtures", params={"team": team_id, "last": last_n})
    if not data or "error" in data:
        return []
    return data.get("response", [])

def get_head_to_head(team_a_id, team_b_id, last_n=10):
    data = api_get("/fixtures", params={"h2h": f"{team_a_id}-{team_b_id}", "last": last_n})
    if not data or "error" in data:
        return []
    return data.get("response", [])

# ---------- AUTO-COMPLETE (Option 2 behavior) ----------
def autocomplete_fill(input_text):
    """Return a canonical team name given partial input. If nothing found, returns input_text stripped."""
    if not input_text or len(input_text.strip()) < 2:
        return input_text.strip()
    candidates = find_team_candidates(input_text, limit=8)
    if candidates:
        # prefer one that startswith the typed text (case-insensitive)
        for c in candidates:
            if c.lower().startswith(input_text.strip().lower()):
                return c
        return candidates[0]
    return input_text.strip()

# ---------- PREDICTION LOGIC ----------
def compute_team_score(team_id):
    stats = {"goals_for":0,"goals_against":0,"wins":0,"draws":0,"losses":0,"matches":0}
    matches = get_recent_matches(team_id, last_n=10)
    if not matches:
        return stats, 0.0
    for m in matches:
        goals = m.get("goals",{}); teams = m.get("teams",{})
        home_id = teams.get("home",{}).get("id"); away_id = teams.get("away",{}).get("id")
        home_goals = goals.get("home"); away_goals = goals.get("away")
        if home_goals is None or away_goals is None:
            continue
        if team_id == home_id:
            gf, ga = home_goals, away_goals
            if home_goals > away_goals: stats["wins"] += 1
            elif home_goals == away_goals: stats["draws"] += 1
            else: stats["losses"] += 1
        else:
            gf, ga = away_goals, home_goals
            if away_goals > home_goals: stats["wins"] += 1
            elif away_goals == home_goals: stats["draws"] += 1
            else: stats["losses"] += 1
        stats["matches"] += 1
        stats["goals_for"] += (gf or 0)
        stats["goals_against"] += (ga or 0)
    form_score = (stats["wins"]*3 + stats["draws"]) / (stats["matches"]*3) if stats["matches"]>0 else 0
    gf_per = stats["goals_for"] / stats["matches"] if stats["matches"]>0 else 0
    score = 0.5 * gf_per + 0.5 * form_score * 3
    return stats, score

def compute_prediction_fixed(team_a, team_b):
    """
    Return stored fixed prediction if exists, else compute using API and save it.
    """
    a = team_a.strip(); b = team_b.strip()
    df = load_data()
    # look for existing record (either order)
    if not df.empty:
        match_row = df[
            ((df['team_a'].str.lower() == a.lower()) & (df['team_b'].str.lower() == b.lower())) |
            ((df['team_a'].str.lower() == b.lower()) & (df['team_b'].str.lower() == a.lower()))
        ]
        if not match_row.empty:
            return match_row.iloc[0].to_dict()

    # auto-complete names then find IDs
    a_name = autocomplete_fill(a)
    b_name = autocomplete_fill(b)
    ta_id, ta_name = find_best_team_id(a_name)
    tb_id, tb_name = find_best_team_id(b_name)
    if not ta_id or not tb_id:
        return {"error": "Could not find one or both teams via API. Try clearer names or check spelling."}

    stats_a, score_a = compute_team_score(ta_id)
    stats_b, score_b = compute_team_score(tb_id)

    # head-to-head
    h2h = get_head_to_head(ta_id, tb_id, last_n=10)
    a_wins=b_wins=draws=0
    if h2h:
        for m in h2h:
            goals = m.get("goals",{}); teams = m.get("teams",{})
            if goals.get("home") is None or goals.get("away") is None:
                continue
            home_id = teams.get("home",{}).get("id"); away_id = teams.get("away",{}).get("id")
            if (home_id == ta_id and goals.get("home")>goals.get("away")) or (away_id == ta_id and goals.get("away")>goals.get("home")):
                a_wins += 1
            elif (home_id == tb_id and goals.get("home")>goals.get("away")) or (away_id == tb_id and goals.get("away")>goals.get("home")):
                b_wins += 1
            else:
                draws += 1
    denom = (a_wins + b_wins + draws)
    h2h_adv = (a_wins - b_wins) / denom if denom>0 else 0

    combined_a = score_a + 0.5 * h2h_adv
    combined_b = score_b - 0.5 * h2h_adv
    diff = combined_a - combined_b
    confidence = float(min(0.99, max(0.01, 0.5 + diff/4)))
    if combined_a > combined_b:
        predicted = a_name
    elif combined_b > combined_a:
        predicted = b_name
    else:
        predicted = "Draw"

    # save fixed prediction
    df = load_data()
    try:
        existing_ids = pd.to_numeric(df['id'], errors='coerce')
        max_id = int(existing_ids.max()) if not existing_ids.empty and not pd.isna(existing_ids.max()) else 0
    except Exception:
        max_id = len(df)
    new_id = int(max_id) + 1
    now = datetime.utcnow().isoformat()
    new_row = {
        "id": new_id, "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "team_a": a_name, "team_b": b_name, "predicted_winner": predicted,
        "prediction_method": "stats_h2h", "confidence": round(confidence,3),
        "actual_result": "", "outcome": "", "notes": f"score_a:{round(combined_a,3)} score_b:{round(combined_b,3)}",
        "created_at": now, "updated_at": now
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df)
    return new_row

# ---------- UI: Pages (tabs) ----------
pages = ["Predictive Hub", "Live Fixtures", "Reviews Book", "Insights Hub"]
page = st.sidebar.selectbox("Go to", pages)

# ---------- Live Fixtures page ----------
if page == "Live Fixtures":
    st.header("📅 Live Fixtures")
    st.markdown("Matches for Today / Tomorrow / Weekend. Click a fixture to auto-fill teams in Predictive Hub.")
    fixtures = []
    try:
        today = date.today()
        # gather today & next 6 days (weekend included)
        for delta in range(0, 7):
            d = today + timedelta(days=delta)
            resp = api_get("/fixtures", params={"date": d.isoformat()})
            if resp and "error" not in resp and resp.get("results", 0) > 0:
                for m in resp.get("response", []):
                    home = m.get("teams", {}).get("home", {}).get("name")
                    away = m.get("teams", {}).get("away", {}).get("name")
                    league = m.get("league", {}).get("name")
                    kickoff = m.get("fixture", {}).get("date", "")[:16].replace("T", " ")
                    fixtures.append({"label": f"{home} vs {away} — {league} ({kickoff})", "home": home, "away": away, "date": d.isoformat()})
    except Exception as e:
        st.error("Error fetching fixtures: " + str(e))

    if not fixtures:
        st.info("No fixtures found in the next 7 days. You can still enter teams manually in Predictive Hub.")
    else:
        sel = st.selectbox("Choose a fixture to analyze", [""] + [f["label"] for f in fixtures])
        if sel:
            chosen = next((x for x in fixtures if x["label"] == sel), None)
            if chosen:
                # store chosen match into session_state so Predictive Hub will pick it up
                st.session_state['selected_home'] = chosen['home']
                st.session_state['selected_away'] = chosen['away']
                st.success(f"Fixture selected: {chosen['home']} vs {chosen['away']}. Now go to Predictive Hub to run analysis.")

# ---------- Predictive Hub page ----------
elif page == "Predictive Hub":
    st.header("⚽ Predictive Hub")
    st.markdown("Use the auto-filled fixture (from Live Fixtures) or type teams (auto-complete fills for you).")

    # prefilled from Live Fixtures
    default_home = st.session_state.get('selected_home', "")
    default_away = st.session_state.get('selected_away', "")

    # manual inputs (auto-complete Option 2: we will overwrite with canonical names when found)
    t1 = st.text_input("Home team", value=default_home, key="home")
    # auto-fill if possible
    try:
        filled1 = autocomplete_fill(t1)
        if filled1 and filled1 != t1:
            st.session_state['home'] = filled1
            t1 = filled1
    except Exception:
        pass

    t2 = st.text_input("Away team", value=default_away, key="away")
    try:
        filled2 = autocomplete_fill(t2)
        if filled2 and filled2 != t2:
            st.session_state['away'] = filled2
            t2 = filled2
    except Exception:
        pass

    if st.button("Get Prediction"):
        if not t1 or not t2:
            st.warning("Please provide both teams (or pick a fixture from Live Fixtures).")
        else:
            with st.spinner("Analyzing..."):
                result = compute_prediction_fixed(t1, t2)
                if isinstance(result, dict) and result.get("error"):
                    st.error(result.get("error"))
                else:
                    st.success(f"Predicted Winner: **{result.get('predicted_winner')}**  (confidence: {result.get('confidence')})")
                    st.markdown("**Details:**")
                    st.write("Method:", result.get("prediction_method"))
                    st.write("Notes:", result.get("notes"))
                    st.write("Saved ID:", result.get("id"))

# ---------- Reviews Book page ----------
elif page == "Reviews Book":
    st.header("📘 Reviews Book")
    st.markdown("All saved predictions. Update with actual results and see your performance chart.")
    df = load_data()
    if df.empty:
        st.info("No predictions yet. Use Predictive Hub to create some.")
    else:
        display_cols = ["id","date","team_a","team_b","predicted_winner","confidence","actual_result","outcome","notes","created_at"]
        st.dataframe(df[display_cols].sort_values("created_at", ascending=False))

        st.markdown("### Update a prediction result")
        sid = st.text_input("Enter prediction ID to update", value="")
        if sid:
            try:
                sid_int = int(sid)
                idx = df.index[df['id'].astype(int) == sid_int]
                if idx.empty:
                    st.error("ID not found.")
                else:
                    i = idx[0]
                    st.write("Selected:", df.at[i, "team_a"], "vs", df.at[i, "team_b"], "| Predicted:", df.at[i, "predicted_winner"])
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
                st.error("Enter a numeric ID.")

        # Download CSV
        try:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("Download predictions CSV", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"⚠️ Error generating CSV: {e}")

        # Accuracy chart
        st.markdown("### Accuracy Chart")
        chart_df = df[df['outcome'].isin(["Correct","Wrong"])]
        if chart_df.empty:
            st.info("No completed results to chart yet.")
        else:
            chart_df['created_at'] = pd.to_datetime(chart_df['created_at'])
            chart_df = chart_df.sort_values('created_at')
            chart_df['is_correct'] = (chart_df['outcome'] == "Correct").astype(int)
            summary = chart_df.groupby(pd.Grouper(key='created_at', freq='D')).agg({'is_correct':'mean','id':'count'}).reset_index()
            summary = summary.rename(columns={'is_correct':'accuracy','id':'matches'})
            fig = px.bar(summary, x='created_at', y='accuracy', labels={'created_at':'Date','accuracy':'Accuracy'}, title='Daily Accuracy (Correct / Total)')
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Insights Hub page ----------
elif page == "Insights Hub":
    st.header("🧠 Insights Hub — Analyst Narrative")
    st.markdown("Type two teams (or pick a fixture on Live Fixtures then go here) and get an analysis.")

    it1 = st.text_input("Team A (full name)", value="")
    it2 = st.text_input("Team B (full name)", value="")
    if st.button("Generate Insights"):
        if not it1 or not it2:
            st.warning("Enter both team names.")
        else:
            with st.spinner("Composing insights..."):
                ta_id, ta_name = find_best_team_id(it1)
                tb_id, tb_name = find_best_team_id(it2)
                if not ta_id or not tb_id:
                    st.error("Could not find one or both teams. Try clearer names.")
                else:
                    ra = get_recent_matches(ta_id, last_n=6)
                    rb = get_recent_matches(tb_id, last_n=6)
                    def summarise(matches, team_id):
                        played=0; wins=draws=losses=0; gf=ga=0
                        for m in matches:
                            goals = m.get("goals",{}); teams = m.get("teams",{})
                            if goals.get("home") is None or goals.get("away") is None: continue
                            home_id = teams.get("home",{}).get("id"); away_id = teams.get("away",{}).get("id")
                            if team_id == home_id:
                                g_for = goals.get("home"); g_against = goals.get("away")
                                if g_for>g_against: wins+=1
                                elif g_for==g_against: draws+=1
                                else: losses+=1
                            else:
                                g_for = goals.get("away"); g_against = goals.get("home")
                                if g_for>g_against: wins+=1
                                elif g_for==g_against: draws+=1
                                else: losses+=1
                            played+=1; gf+= (g_for or 0); ga += (g_against or 0)
                        return {"played":played,"wins":wins,"draws":draws,"losses":losses,"gf":gf,"ga":ga}

                    sa = summarise(ra, ta_id); sb = summarise(rb, tb_id)
                    h2h = get_head_to_head(ta_id, tb_id, last_n=6)
                    h2h_summary = {"a_wins":0,"b_wins":0,"draws":0}
                    for m in h2h:
                        goals = m.get("goals",{})
                        if goals.get("home") is None or goals.get("away") is None: continue
                        home = m.get("teams",{}).get("home",{}).get("id"); away = m.get("teams",{}).get("away",{}).get("id")
                        if (home==ta_id and goals.get("home")>goals.get("away")) or (away==ta_id and goals.get("away")>goals.get("home")):
                            h2h_summary["a_wins"] += 1
                        elif (home==tb_id and goals.get("home")>goals.get("away")) or (away==tb_id and goals.get("away")>goals.get("home")):
                            h2h_summary["b_wins"] += 1
                        else:
                            h2h_summary["draws"] += 1

                    narrative = []
                    narrative.append(f"**Overview:** {ta_name} vs {tb_name}.")
                    narrative.append(f"{ta_name} recent form: {sa['wins']}W {sa['draws']}D {sa['losses']}L across {sa['played']} matches. Goals F/A: {sa['gf']}/{sa['ga']}.")
                    narrative.append(f"{tb_name} recent form: {sb['wins']}W {sb['draws']}D {sb['losses']}L across {sb['played']} matches. Goals F/A: {sb['gf']}/{sb['ga']}.")
                    if h2h:
                        narrative.append(f"Head-to-head (last {len(h2h)}): {h2h_summary['a_wins']} wins for {ta_name}, {h2h_summary['b_wins']} wins for {tb_name}, {h2h_summary['draws']} draws.")
                    # heuristic prediction
                    score_a = (sa['wins']*3 + sa['draws'])/max(1,sa['played']) if sa['played']>0 else 0
                    score_b = (sb['wins']*3 + sb['draws'])/max(1,sb['played']) if sb['played']>0 else 0
                    score_a += (sa['gf'] - sa['ga'])/max(1, sa['played']) if sa['played']>0 else 0
                    score_b += (sb['gf'] - sb['ga'])/max(1, sb['played']) if sb['played']>0 else 0
                    score_a += h2h_summary['a_wins']*0.5
                    score_b += h2h_summary['b_wins']*0.5
                    if score_a > score_b: pred = ta_name
                    elif score_b > score_a: pred = tb_name
                    else: pred = "Draw"
                    conf = min(0.99, max(0.05, 0.5 + (abs(score_a-score_b)/4)))
                    narrative.append(f"**Model says:** {pred} is more likely (confidence {conf*100:.1f}%).")
                    st.markdown("\n\n".join(narrative))

# ---------- Footer ----------
st.markdown("---")
st.caption("PredictX Pro Ultra V3 — keep your API key private.")
