
import streamlit as st
import random

st.set_page_config(page_title="PredictX Pro Ultra V2", layout="centered")

st.title("⚽ PredictX Pro Ultra V2")
st.write("Welcome to your AI-powered football prediction simulator!")

# Input fields
team1 = st.text_input("Enter Team A")
team2 = st.text_input("Enter Team B")

if st.button("Predict"):
    if team1 and team2:
        winner = random.choice([team1, team2])
        st.success(f"🏆 Predicted Winner: **{winner}**")
    else:
        st.warning("Please enter both team names.")

st.markdown("---")
st.caption("Powered by PredictX AI ⚡")
