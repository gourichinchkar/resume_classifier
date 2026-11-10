import streamlit as st
from Step6_Match_scoring  import analyze

st.set_page_config(page_title="Resume Classifier & Job Matcher", layout="centered")
st.title("Resume Classifier & Job Match Score")
st.write("Resume ani job description paste kara. App resume cha category predict karel ani match % dakhavel.")

resume = st.text_area("Resume text", height=220)
jd = st.text_area("Job description", height=220)

if st.button("Analyze"):
    if not resume.strip() or not jd.strip():
        st.warning("Donhi boxes bhara.")
    else:
        category, score, missing = analyze(resume, jd)
        st.subheader("Result")
        st.metric("Predicted category", category)
        st.metric("Match score", f"{score}%")
        st.progress(min(int(score), 100))
        if missing:
            st.write("**Job description madhle keywords je resume madhe nahit:**")
            st.write(", ".join(missing))
        else:
            st.success("Important keywords sagle resume madhe aahet.")