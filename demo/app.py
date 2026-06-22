import streamlit as st
import json
import pandas as pd
import sys
import os
import time

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.jd_parser            import parse_job_description
from src.data_processor       import check_is_honeypot, check_is_consulting_disqualified, stream_candidates
from src.feature_engineer     import extract_features
from src.scorer               import calculate_composite_score
from src.reasoning_generator  import generate_reasoning

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recruiter Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero { font-size: 2.2rem; font-weight: 700;
  background: linear-gradient(135deg,#667eea,#764ba2,#f64f59);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color: #9ca3af; font-size: 1rem; margin-top: -6px; margin-bottom: 18px; }

.card {
  background: linear-gradient(135deg,#1e2230,#252d3a);
  border-radius: 12px; padding: 18px; border: 1px solid #2e3440;
  text-align: center; margin-bottom: 8px;
}
.card h4 { color: #9ca3af; font-size: 0.72rem; letter-spacing: 1px;
           text-transform: uppercase; margin: 0 0 6px 0; }
.card h2 { color: #fff; font-size: 1.9rem; font-weight: 700; margin: 0; }

.rank-gold   { background: linear-gradient(135deg,#78350f,#92400e); border-color:#d97706; }
.rank-silver { background: linear-gradient(135deg,#1f2937,#374151); border-color:#9ca3af; }
.rank-bronze { background: linear-gradient(135deg,#1c1917,#292524); border-color:#c2956c; }

.section-hdr {
  font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
  border-left: 3px solid #667eea; padding-left: 10px;
  margin: 20px 0 10px 0;
}
.tag { display:inline-block; border-radius:6px; padding:3px 9px;
       font-size:.73rem; margin:2px; }
.tag-core { background:rgba(102,126,234,.2); color:#818cf8; border:1px solid #4f46e5; }
.tag-pref { background:rgba(16,185,129,.15); color:#6ee7b7; border:1px solid #059669; }

.stButton>button {
  background: linear-gradient(135deg,#667eea,#764ba2);
  color:white; border:none; padding:11px 28px;
  font-weight:600; border-radius:8px; width:100%;
}
.stButton>button:hover { opacity:.88; }
.stRadio > div { gap: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero'>🧠 General AI Recruiter Brain</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Upload any JD + candidates → dynamic criteria extraction → filter → score → ranked shortlist with reasoning</div>", unsafe_allow_html=True)
st.markdown("---")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline")
    st.info(
        "**1. Parse JD** — extract YOE, locations, skills, consulting gate\n\n"
        "**2. Filter** — remove honeypots + JD-driven consulting disqualification\n\n"
        "**3. Score** — skill match × behavioral multiplier + location + notice\n\n"
        "**4. Rank** — sort by composite score, assign ranks 1–N\n\n"
        "**5. Reason** — generate fact-grounded 1-2 sentence justification"
    )
    st.markdown("---")
    st.caption("Scoring weights")
    st.markdown("""
| Signal | Weight |
|---|---|
| Core skill match | 45% |
| YOE fit | 15% |
| Preferred skills | 10% |
| Location | 10% |
| Behavioral ×multiplier | 0.70–1.30× |
    """)
    st.caption("Behavioral signals act as a multiplier (per redrob_signals_doc.txt), not additive.")

# ─── STEP 1: JD ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Step 1 — Job Description</div>", unsafe_allow_html=True)

jd_source = st.radio(
    "Choose JD source:",
    ["Use sample Redrob JD", "Upload a JD file (.txt / .md)"],
    horizontal=True, key="jd_source"
)

jd_text  = None
criteria = None

if jd_source == "Use sample Redrob JD":
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "India_runs_data_and_ai_challenge", "job_description.txt"
    )
    if os.path.exists(path):
        jd_text = open(path, encoding="utf-8").read()
        st.success("✅ Sample Redrob JD loaded")
    else:
        st.error("Sample JD not found at expected path.")

else:  # Upload
    jd_file = st.file_uploader("Upload JD file", type=["txt", "md"], key="jd_file")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
        st.success(f"✅ Loaded: **{jd_file.name}**")

if jd_text:
    criteria = parse_job_description(jd_text)
    with st.expander("📋 Extracted JD Criteria", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='card'><h4>Experience Range</h4><h2>{criteria.min_yoe:.0f}–{criteria.max_yoe:.0f} yrs</h2></div>", unsafe_allow_html=True)
        with c2:
            locs = ", ".join(sorted(criteria.preferred_locations)) or "Any"
            st.markdown(f"<div class='card'><h4>Preferred Locations</h4><h2 style='font-size:1rem;margin-top:10px'>{locs}</h2></div>", unsafe_allow_html=True)
        with c3:
            if criteria.disallow_consulting:
                cl, cc = "❌ Consulting Excluded", "#ef4444"
            else:
                cl, cc = "✅ Consulting Allowed", "#10b981"
            st.markdown(f"<div class='card'><h4>Consulting Filter</h4><h2 style='font-size:.85rem;color:{cc};margin-top:8px'>{cl}</h2></div>", unsafe_allow_html=True)

        st.markdown("**Core Required Skills (from JD):**")
        st.markdown("".join(f"<span class='tag tag-core'>{s}</span>" for s in sorted(criteria.core_skills)), unsafe_allow_html=True)
        st.markdown("**Preferred Skills (from JD):**")
        st.markdown("".join(f"<span class='tag tag-pref'>{s}</span>" for s in sorted(criteria.preferred_skills)), unsafe_allow_html=True)
        if criteria.tier1_locations:
            st.caption(f"Tier-1 cities also accepted: {', '.join(sorted(criteria.tier1_locations))}")

st.markdown("---")

# ─── STEP 2: CANDIDATES ───────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Step 2 — Candidate Pool</div>", unsafe_allow_html=True)

cand_source = st.radio(
    "Choose candidate source:",
    ["Use sample candidates (50 profiles)", "Upload a file (.json / .jsonl)", "Use local file path (for large files like candidates.jsonl)"],
    horizontal=True, key="cand_source"
)

# We'll distinguish between pre-loaded list vs a path to stream from
raw_candidates  = []   # for small uploads / sample
cand_file_path  = None # for large local files — streamed directly
cand_file       = None

if cand_source == "Use sample candidates (50 profiles)":
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "India_runs_data_and_ai_challenge", "sample_candidates.json"
    )
    if os.path.exists(path):
        raw_candidates = json.load(open(path, encoding="utf-8"))
        st.success(f"✅ Loaded {len(raw_candidates)} sample candidates")
    else:
        st.error("Sample candidates file not found.")

elif cand_source == "Upload a file (.json / .jsonl)":
    cand_file = st.file_uploader(
        "Upload candidates file",
        type=["json", "jsonl"],
        key="cand_file"
    )
    if cand_file:
        size_mb = cand_file.size / (1024 * 1024)
        st.success(f"✅ Uploaded file: **{cand_file.name}** ({size_mb:.2f} MB) — ready for streaming")

else:  # Local file path
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "India_runs_data_and_ai_challenge", "candidates.jsonl"
    )
    local_path = st.text_input(
        "Enter absolute path to candidates.jsonl:",
        value=default_path,
        key="local_path"
    )
    if local_path and os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        st.success(f"✅ File found: **{os.path.basename(local_path)}** ({size_mb:.0f} MB) — will be streamed directly")
        cand_file_path = local_path
    elif local_path:
        st.error(f"File not found: `{local_path}`")

st.markdown("---")

# ─── STEP 3: RUN ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Step 3 — Run Full Pipeline</div>", unsafe_allow_html=True)

ready = criteria and (
    (cand_source == "Use sample candidates (50 profiles)" and len(raw_candidates) > 0) or
    (cand_source == "Upload a file (.json / .jsonl)" and cand_file is not None) or
    (cand_source == "Use local file path (for large files like candidates.jsonl)" and cand_file_path is not None)
)

if not criteria:
    st.info("👆 Complete Step 1 — select or upload a Job Description.")
elif not ready:
    st.info("👆 Complete Step 2 — select or provide a candidate pool.")

if ready:
    top_n = st.slider("How many top candidates to rank?", min_value=10, max_value=100, value=100, step=10)

    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None

    if st.button("🚀 Run Ranking Pipeline", key="run_btn"):
        t0 = time.time()

        valid, honeypots, consulting_out = [], [], []
        is_streaming = False
        candidates_to_process = []

        if cand_source == "Use sample candidates (50 profiles)":
            candidates_to_process = raw_candidates
        elif cand_source == "Use local file path (for large files like candidates.jsonl)":
            is_streaming = True
        elif cand_source == "Upload a file (.json / .jsonl)":
            cand_file.seek(0)
            first_byte = cand_file.read(1)
            cand_file.seek(0)
            if first_byte == b"[":
                content = cand_file.read().decode("utf-8")
                try:
                    candidates_to_process = json.loads(content)
                except Exception as e:
                    st.error(f"Error parsing uploaded JSON array: {e}")
                    candidates_to_process = []
            else:
                is_streaming = True

        # ── Stream or iterate ──────────────────────────────────────────────
        if is_streaming:
            progress = st.progress(0, text="Streaming and filtering candidates...")
            count = 0

            if cand_source == "Use local file path (for large files like candidates.jsonl)":
                # First pass: count total lines for progress
                with open(cand_file_path) as f:
                    total_lines = sum(1 for _ in f)

                with open(cand_file_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        count += 1
                        try:
                            cand = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        is_hp, hp_r = check_is_honeypot(cand)
                        if is_hp:
                            honeypots.append({"candidate_id": cand["candidate_id"],
                                              "name": cand["profile"]["anonymized_name"],
                                              "reason": hp_r})
                            continue
                        if criteria.disallow_consulting:
                            is_cd, cd_r = check_is_consulting_disqualified(cand)
                            if is_cd:
                                consulting_out.append({"candidate_id": cand["candidate_id"],
                                                       "name": cand["profile"]["anonymized_name"],
                                                       "reason": cd_r})
                                continue
                        valid.append(cand)

                        if count % 5000 == 0:
                            pct = min(count / total_lines, 1.0)
                            progress.progress(pct, text=f"Streamed {count:,}/{total_lines:,} candidates... ({len(valid):,} valid so far)")
            else:
                # Stream from uploaded file
                cand_file.seek(0)
                total_lines = sum(1 for _ in cand_file)
                cand_file.seek(0)

                for line_bytes in cand_file:
                    line = line_bytes.decode("utf-8")
                    if not line.strip():
                        continue
                    count += 1
                    try:
                        cand = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    is_hp, hp_r = check_is_honeypot(cand)
                    if is_hp:
                        honeypots.append({"candidate_id": cand["candidate_id"],
                                          "name": cand["profile"]["anonymized_name"],
                                          "reason": hp_r})
                        continue
                    if criteria.disallow_consulting:
                        is_cd, cd_r = check_is_consulting_disqualified(cand)
                        if is_cd:
                            consulting_out.append({"candidate_id": cand["candidate_id"],
                                                   "name": cand["profile"]["anonymized_name"],
                                                   "reason": cd_r})
                            continue
                    valid.append(cand)

                    if count % 5000 == 0:
                        pct = min(count / total_lines, 1.0)
                        progress.progress(pct, text=f"Streamed {count:,}/{total_lines:,} candidates... ({len(valid):,} valid so far)")

            progress.progress(1.0, text=f"✅ Done — {len(valid):,} valid candidates from {count:,} total")
            total_input = count

        else:
            # Small file / sample / JSON list: direct loop
            for cand in candidates_to_process:
                is_hp, hp_r = check_is_honeypot(cand)
                if is_hp:
                    honeypots.append({"candidate_id": cand["candidate_id"],
                                      "name": cand["profile"]["anonymized_name"],
                                      "reason": hp_r})
                    continue
                if criteria.disallow_consulting:
                    is_cd, cd_r = check_is_consulting_disqualified(cand)
                    if is_cd:
                        consulting_out.append({"candidate_id": cand["candidate_id"],
                                               "name": cand["profile"]["anonymized_name"],
                                               "reason": cd_r})
                        continue
                valid.append(cand)
            total_input = len(candidates_to_process)

        # ── Score all valid candidates ─────────────────────────────────────
        score_progress = st.progress(0, text="Scoring candidates...")
        scored = []
        for i, cand in enumerate(valid):
            feats             = extract_features(cand, criteria)
            score, components = calculate_composite_score(cand, feats)
            scored.append({"cand": cand, "feats": feats,
                           "components": components, "score": score})
            if i % 1000 == 0:
                score_progress.progress(min(i / max(len(valid), 1), 1.0),
                                        text=f"Scored {i:,}/{len(valid):,}...")
        score_progress.progress(1.0, text="✅ Scoring complete")

        # ── Sort & rank ────────────────────────────────────────────────────
        scored.sort(key=lambda x: (-x["score"], x["cand"]["candidate_id"]))
        top_scored = scored[:top_n]

        # ── Reasoning ─────────────────────────────────────────────────────
        rows = []
        for rank_idx, item in enumerate(top_scored, 1):
            reasoning = generate_reasoning(
                cand=item["cand"], rank=rank_idx,
                features=item["feats"], components=item["components"],
                criteria=criteria,
            )
            p = item["cand"]["profile"]
            rows.append({
                "rank":         rank_idx,
                "candidate_id": item["cand"]["candidate_id"],
                "name":         p.get("anonymized_name", ""),
                "title":        p.get("current_title", ""),
                "company":      p.get("current_company", ""),
                "yoe":          p.get("years_of_experience", 0),
                "location":     p.get("location", ""),
                "score":        item["score"],
                "core_%":       f"{item['feats']['core_skill_ratio']*100:.0f}%",
                "beh_×":        f"{item['components']['beh_multiplier']:.2f}×",
                "notice_d":     item["feats"]["notice_days"],
                "open_work":    "✅" if item["feats"]["open_to_work"] else "—",
                "reasoning":    reasoning,
                "redflags":     "; ".join(item["feats"].get("redflags", [])) or "—",
            })

        elapsed = time.time() - t0

        # Save to session state
        st.session_state.pipeline_results = {
            "total_input": total_input,
            "valid_len": len(valid),
            "honeypots": honeypots,
            "consulting_out": consulting_out,
            "elapsed": elapsed,
            "disallow_consulting": criteria.disallow_consulting,
            "rows": rows,
        }

    # Render results from session state (persists across download interactions)
    if st.session_state.pipeline_results is not None:
        res = st.session_state.pipeline_results
        total_input = res["total_input"]
        valid_len = res["valid_len"]
        honeypots = res["honeypots"]
        consulting_out = res["consulting_out"]
        elapsed = res["elapsed"]
        disallow_consulting = res["disallow_consulting"]
        rows = res["rows"]

        # ── Stats row ─────────────────────────────────────────────────────
        st.markdown("#### 📊 Pipeline Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"<div class='card'><h4>Total Input</h4><h2>{total_input:,}</h2></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='card'><h4>Valid & Scored</h4><h2 style='color:#10b981'>{valid_len:,}</h2></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='card' style='border-color:#ef4444'><h4>Honeypots ❌</h4><h2 style='color:#ef4444'>{len(honeypots):,}</h2></div>", unsafe_allow_html=True)
        with c4:
            label = f"{len(consulting_out):,}" if disallow_consulting else "N/A"
            color = "#f59e0b" if disallow_consulting else "#6b7280"
            st.markdown(f"<div class='card' style='border-color:{color}'><h4>Consulting ❌</h4><h2 style='color:{color}'>{label}</h2></div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div class='card'><h4>Runtime</h4><h2>{elapsed:.1f}s</h2></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Podium ────────────────────────────────────────────────────────
        if rows:
            st.markdown("#### 🏆 Top 3 Candidates")
            p1, p2, p3 = st.columns(3)
            for col, idx, cls in [(p1, 0, "rank-gold"), (p2, 1, "rank-silver"), (p3, 2, "rank-bronze")]:
                if idx < len(rows):
                    r = rows[idx]
                    with col:
                        st.markdown(f"""
                        <div class='card {cls}'>
                          <h4>#{r['rank']}</h4>
                          <h2 style='font-size:1rem'>{r['name']}</h2>
                          <p style='color:#d1d5db;font-size:.8rem;margin:4px 0'>{r['title']} @ {r['company']}</p>
                          <p style='color:#a5b4fc;font-size:.88rem;margin:0'>Score: {r['score']:.4f}</p>
                          <p style='color:#9ca3af;font-size:.75rem;margin:4px 0'>{r['yoe']} yrs · {r['location']}</p>
                          <p style='color:#9ca3af;font-size:.72rem;margin:0'>Core: {r['core_%']} · Beh: {r['beh_×']} · Notice: {r['notice_d']}d</p>
                        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Full ranked table with tabs ────────────────────────────────────
        st.markdown(f"#### 📋 Full Ranked Results (Top {len(rows)})")
        df = pd.DataFrame(rows)

        tab_main, tab_flags, tab_filter = st.tabs(["🏅 Ranked Results", "⚠️ Red Flags", "🚫 Filtered Out"])

        with tab_main:
            show_cols = ["rank", "candidate_id", "name", "title", "company",
                         "yoe", "location", "score", "core_%", "beh_×",
                         "notice_d", "open_work", "reasoning"]
            st.dataframe(df[show_cols], use_container_width=True, height=500)
            csv_out = df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
            st.download_button(
                label="⬇️ Download Submission CSV",
                data=csv_out,
                file_name="submission.csv",
                mime="text/csv",
                key="download_csv"
            )

        with tab_flags:
            flagged = df[df["redflags"] != "—"][["rank", "candidate_id", "name", "title", "score", "redflags"]]
            if not flagged.empty:
                st.dataframe(flagged, use_container_width=True)
            else:
                st.info("No red flags detected in this pool.")

        with tab_filter:
            if honeypots:
                st.markdown(f"**🚨 {len(honeypots)} Honeypots Removed**")
                st.dataframe(pd.DataFrame(honeypots), use_container_width=True)
            if consulting_out:
                st.markdown(f"**⚠️ {len(consulting_out)} Consulting-Disqualified (JD gate was active)**")
                st.dataframe(pd.DataFrame(consulting_out), use_container_width=True)
            if not honeypots and not consulting_out:
                st.info("No candidates were filtered out.")
            if not disallow_consulting:
                st.info("ℹ️ Consulting filter was OFF for this JD — all consulting backgrounds passed through.")
