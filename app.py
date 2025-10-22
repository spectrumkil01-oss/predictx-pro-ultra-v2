# app.py - PredictX Pro Ultra V3 (Final)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import io
from datetime import datetime, timedelta, date
import plotly.express as px
from difflib import get_close_matches

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PredictX Pro Ultra V3", layout="wide", page_icon="⚽")
DATA_FILE = "predictions.csv"

# <<< YOUR API-FOOTBALL KEY (keep private) >>>
API_KEY = "2debb76dcc5bbe808e64d70de9b17abf"
API_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# ---------------- STYLE (Dark) ----------------
st.markdown(
    """
    <style>
    :root{ --card-bg: rgba(10,14,23,0.85); --muted:#94a3b8; --accent:#2da8ff; }
    body { background-color: #020617; color: #dbeafe; }
    .stApp { background: linear-gradient(180deg, rgba(2,6,12,0.95), rgba(4,10,20,0.95)); }
    .card { background: var(--card-bg); padding: 14px; border-radius: 12px; box-shadow: 0 6px 30px rgba(2,6,23,0.6); }
    .title { font-size:22px; font-weight:700; color:#e6f6ff; }
    footer { color: #8298a6; text-align:center; padding:8px; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">⚽ PredictX Pro Ultra V3</div>', unsafe_allow_html=True)
st.caption("AI-Powered Football Prediction Analyzer — Predictive Hub & Review System")

# ---------------- DATA HELPERS ----------------
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

# ---------------- API HELPERS ----------------
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

def find_team_candidates(query, limit=5):
    """Return list of candidate team names from API search"""
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
    """Try to find the best matching team ID and canonical name globally"""
    # 1) direct API search
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

# ---------------- AUTO-COMPLETE (Option 2) ----------------
def autocomplete_fill(input_text):
    """
    Given a partial team name typed by user, try to find a canonical name and return it.
    Uses API search and difflib fallback if needed.
    """
    if not input_text or len(input_text.strip()) < 2:
        return input_text.strip()
    # try API search candidates
    candidates = find_team_candidates(input_text, limit=10)
    if candidates:
        # try exact case-insensitive match
        for c in candidates:
            if c.lower().startswith(input_text.strip().lower()):
                return c
        # else return top candidate
        return candidates[0]
    # fallback: return original if nothing found
    return input_text.strip()

# ---------------- PREDICTION LOGIC ----------------
def compute_team_score(team_id):
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
    form_score = (stats["wins"]*3 + stats["draws"]) / (stats["matches"]*3) if stats["matches"]>0 else 0
    gf_per = stats["goals_for"] / stats["matches"] if stats["matches"]>0 else 0
    score = 0.5 * gf_per + 0.5 * form_score * 3
    return stats, score

def compute_prediction_fixed(team_a, team_b):
    """
    Return stored fixed prediction if exists; otherwise compute with live data and store.
    Returns dict (stored/new row) or dict with 'error' key on failure.
    """
    # normalize
    a = team_a.strip()
    b = team_b.strip()
    df = load_data()

    # check stored both-order
    if not df.empty:
        match_row = df[
            ((df['team_a'].str.lower() == a.lower()) & (df['team_b'].str.lower() == b.lower())) |
            ((df['team_a'].str.lower() == b.lower()) & (df['team_b'].str.lower() == a.lower()))
        ]
        if not match_row.empty:
            return match_row.iloc[0].to_dict()

    # try to find canonical teams via API (auto-complete then id lookup)
    a_name = autocomplete_fill(a)
    b_name = autocomplete_fill(b)
    ta_id, ta_name = find_best_team_id(a_name)
    tb_id, tb_name = find_best_team_id(b_name)
    if not ta_id or not tb_id:
        return {"error": "Could not find one or both teams via API. Try clearer names or check spelling."}

    # compute stats
    stats_a, score_a = compute_team_score(ta_id)
    stats_b, score_b = compute_team_score(tb_id)

    # head-to-head
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
    denom = (a_wins + b_wins + draws)
    h2h_adv = (a_wins - b_wins) / denom if denom > 0 else 0

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

    # prepare row & save
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
        "team_a": a_name,
        "team_b": b_name,
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

# ---------------- UI TABS ----------------
tabs = st.tabs(["⚽ Predictive Hub", "📘 Reviews Book", "📝 Update Results", "🧠 Insights Hub"])

# ---------------- Predictive Hub ----------------
with tabs[0]:
    st.header("⚽ Predictive Hub — Live Matches & Auto-complete")
    st.markdown("Pick today's/tomorrow's/weekend fixtures or type team names. The app auto-fills team names for you.")

    # --- Fixtures: Today / Tomorrow / Weekend ---
    col_f1, col_f2 = st.columns([2,3])
    with col_f1:
        st.subheader("📅 Upcoming Matches (Today / Tomorrow / Weekend)")
        # collect fixtures for today, tomorrow, weekend (Fri-Sun)
        fixtures = []
        try:
            today_date = date.today()
            # today
            for delta in range(0, 3):  # today + next 2 days
                d = today_date + timedelta(days=delta)
                resp = api_get("/fixtures", params={"date": d.isoformat()})
                if resp and "error" not in resp and resp.get("results", 0) > 0:
                    for m in resp.get("response", []):
                        home = m.get("teams", {}).get("home", {}).get("name")
                        away = m.get("teams", {}).get("away", {}).get("name")
                        league = m.get("league", {}).get("name")
                        kickoff = m.get("fixture", {}).get("date", "")[:16].replace("T", " ")
                        fixtures.append({"label": f"{home} vs {away} — {league} ({kickoff})", "home": home, "away": away})
        except Exception as e:
            st.warning("Fixtures endpoint error: " + str(e))

        if fixtures:
            sel_fixture = st.selectbox("Select a fixture (auto-fill teams)", [""] + [f["label"] for f in fixtures])
            if sel_fixture:
                chosen = next((x for x in fixtures if x["label"] == sel_fixture), None)
                if chosen:
                    auto_home = chosen["home"]
                    auto_away = chosen["away"]
                    st.success(f"Selected: {auto_home} vs {auto_away}")
                else:
                    auto_home = ""
                    auto_away = ""
        else:
            st.info("No upcoming fixtures found for the next 3 days. Try typing team names below.")
            auto_home = ""
            auto_away = ""

    with col_f2:
        st.subheader("🔎 Type teams (auto-complete will fill)")
        # auto-complete textboxes: Option 2 behaviour (fill automatically to canonical name)
        # Home team input
        default_home = auto_home if 'auto_home' in locals() else ""
        t1 = st.text_input("Home team", value=default_home, key="home_input")
        # try fill
        try:
            filled1 = autocomplete_fill(t1)
            if filled1 and filled1 != t1:
                # automatically set filled value (Option 2)
                st.session_state.home_input = filled1
                t1 = filled1
        except Exception:
            pass

        # Away team input
        default_away = auto_away if 'auto_away' in locals() else ""
        t2 = st.text_input("Away team", value=default_away, key="away_input")
        try:
            filled2 = autocomplete_fill(t2)
            if filled2 and filled2 != t2:
                st.session_state.away_input = filled2
                t2 = filled2
        except Exception:
            pass

        # Buttons
        if st.button("Get Prediction"):
            if not t1 or not t2:
                st.warning("Please enter both teams (or pick a fixture).")
            else:
                with st.spinner("Computing prediction..."):
                    res = compute_prediction_fixed(t1, t2)
                    if isinstance(res, dict) and res.get("error"):
                        st.error(res.get("error"))
                    else:
                        st.success(f"Predicted Winner: **{res.get('predicted_winner')}** (confidence: {res.get('confidence')})")
                        st.markdown("**Details:**")
                        st.write("Method:", res.get("prediction_method"))
                        st.write("Notes:", res.get("notes"))
                        st.write("Saved ID:", res.get("id"))

# ---------------- Reviews Book ----------------
with tabs[1]:
    st.header("📘 Reviews Book — History & Accuracy")
    df = load_data()
    if df.empty:
        st.info("No predictions yet. Make one in Predictive Hub.")
    else:
        df_display = df.copy()
        display_cols = ["id","date","team_a","team_b","predicted_winner","confidence","actual_result","outcome","notes","created_at"]
        st.dataframe(df_display[display_cols].sort_values("created_at", ascending=False))

        st.markdown("### Update a prediction manually")
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
                    actual = st.text_input("Enter actual score or winner (e.g. 2-1 or Draw)", value=df.at[i, "actual_result"])
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
            st.info("No completed results to chart yet. Mark outcomes to see accuracy.")
        else:
            chart_df['created_at'] = pd.to_datetime(chart_df['created_at'])
            chart_df = chart_df.sort_values('created_at')
            chart_df['is_correct'] = (chart_df['outcome'] == "Correct").astype(int)
            summary = chart_df.groupby(pd.Grouper(key='created_at', freq='D')).agg({'is_correct':'mean','id':'count'}).reset_index()
            summary = summary.rename(columns={'is_correct':'accuracy','id':'matches'})
            fig = px.bar(summary, x='created_at', y='accuracy', labels={'created_at':'Date','accuracy':'Accuracy'}, title='Daily Accuracy (Correct / Total)')
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Update Results (auto) ----------------
with tabs[2]:
    st.header("📝 Update Results — Auto-fetch Live Scores")
    df = load_data()
    if df.empty:
        st.info("No predictions yet.")
    else:
        choices = df.index.tolist()
        def choice_label(i):
            r = df.loc[i]
            return f"[{r['id']}] {r['team_a']} vs {r['team_b']} ({r['date']}) - {r.get('outcome','') or 'Pending'}"
        sel = st.selectbox("Select prediction to auto-check:", choices, format_func=choice_label)
        if sel is not None:
            team_a = df.loc[sel, 'team_a']
            team_b = df.loc[sel, 'team_b']
            st.write("Selected:", team_a, "vs", team_b)
            if st.button("🔍 Fetch live result"):
                try:
                    # find team ids
                    ta_id, _ = find_best_team_id(team_a)
                    tb_id, _ = find_best_team_id(team_b)
                    if not ta_id or not tb_id:
                        st.error("Could not find teams by API IDs for auto-fetch.")
                    else:
                        saved_date = df.loc[sel, 'date']
                        url = f"{API_BASE}/fixtures?team={ta_id}&date={saved_date}"
                        resp = requests.get(url, headers=HEADERS, timeout=12).json()
                        fixtures = resp.get("response", [])
                        found = None
                        for fx in fixtures:
                            home = fx['teams']['home']['name']
                            away = fx['teams']['away']['name']
                            if (team_a.lower() in home.lower() and team_b.lower() in away.lower()) or (team_a.lower() in away.lower() and team_b.lower() in home.lower()):
                                found = fx
                                break
                        if not found:
                            st.error("Could not find matching fixture for that date. Try manual update.")
                        else:
                            goals = found.get("goals", {})
                            home_goals = goals.get("home")
                            away_goals = goals.get("away")
                            if home_goals is None or away_goals is None:
                                st.info("Match not finished or score not yet available.")
                            else:
                                if home_goals > away_goals:
                                    actual = found['teams']['home']['name']
                                elif away_goals > home_goals:
                                    actual = found['teams']['away']['name']
                                else:
                                    actual = "Draw"
                                df.at[sel, 'actual_result'] = f"{home_goals}-{away_goals}"
                                df.at[sel, 'outcome'] = "Correct" if df.at[sel, 'predicted_winner'] == actual else "Wrong"
                                df.at[sel, 'updated_at'] = datetime.utcnow().isoformat()
                                save_data(df)
                                st.success(f"Updated: {actual} ({home_goals}-{away_goals}). Outcome set to {df.at[sel,'outcome']}")
                                st.experimental_rerun()
                except Exception as e:
                    st.error("Error while fetching fixture: " + str(e))

# ---------------- Insights Hub (Narrative) ----------------
with tabs[3]:
    st.header("🧠 Insights Hub — Analyst Narrative")
    st.markdown("Type or pick two teams and generate an analyst-style writeup (recent form, attack/defense, H2H).")

    c1, c2 = st.columns(2)
    with c1:
        it1 = st.text_input("Team A (full name)", value="", key="ins_a")
    with c2:
        it2 = st.text_input("Team B (full name)", value="", key="ins_b")

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
                        played=0; wins=draws=losses=0; gf=ga=0; rows=[]
                        for m in matches:
                            goals = m.get("goals",{}); teams = m.get("teams",{})
                            if goals.get("home") is None or goals.get("away") is None:
                                continue
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
                            rows.append((m.get("fixture",{}).get("date","")[:10], g_for, g_against))
                        return {"played":played,"wins":wins,"draws":draws,"losses":losses,"gf":gf,"ga":ga,"rows":rows}

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
                    # heuristic
                    score_a = (sa['wins']*3 + sa['draws'])/max(1,sa['played']) if sa['played']>0 else 0
                    score_b = (sb['wins']*3 + sb['draws'])/max(1,sb['played']) if sb['played']>0 else 0
                    score_a += (sa['gf'] - sa['ga'])/max(1, sa['played']) if sa['played']>0 else 0
                    score_b += (sb['gf'] - sb['ga'])/max(1, sb['played']) if sb['played']>0 else 0
                    score_a += h2h_summary['a_wins']*0.5
                    score_b += h2h_summary['b_wins']*0.5
                    if score_a > score_b:
                        pred = ta_name
                    elif score_b > score_a:
                        pred = tb_name
                    else:
                        pred = "Draw"
                    conf = min(0.99, max(0.05, 0.5 + (abs(score_a-score_b)/4)))
                    narrative.append(f"**Model says:** {pred} is more likely (confidence {conf*100:.1f}%).")
                    tips=[]
                    if sa['played'] < 3: tips.append(f"{ta_name} has limited recent matches — be cautious.")
                    if sb['played'] < 3: tips.append(f"{tb_name} has limited recent matches — be cautious.")
                    if sa['gf'] > sb['ga'] + 2: tips.append(f"{ta_name} shows attacking edge.")
                    if sb['gf'] > sa['ga'] + 2: tips.append(f"{tb_name} shows attacking edge.")

                    st.markdown("\n\n".join(narrative))
                    if tips:
                        st.markdown("**Quick analyst tips:**")
                        for t in tips:
                            st.markdown(f"- {t}")

# -------------- footer --------------
st.markdown("---")
st.caption("PredictX Pro Ultra V3 — keep your API key private. Built with API-FOOTBALL.")
