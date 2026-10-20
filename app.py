
import streamlit as st
import pandas as pd
import joblib
import re
import html
from pathlib import Path
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="ResumeMatch AI",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resume_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

st.markdown("""
<style>
.stApp {
    background: #ffffff !important;
    color: #202124 !important;
}

header[data-testid="stHeader"] {
    background: #ffffff !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, p, label {
    color: #202124 !important;
}

/* Clean job description field */
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #202124 !important;
    border: 1px solid #d5dbe5 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    outline: none !important;
    caret-color: #202124 !important;
}

div[data-testid="stTextArea"] textarea:focus {
    background: #ffffff !important;
    color: #202124 !important;
    border: 1px solid #9aa9c2 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Clean uploader */
div[data-testid="stFileUploader"] {
    background: #ffffff !important;
    color: #202124 !important;
    border: 1px solid #d5dbe5 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    box-shadow: none !important;
}

div[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    color: #202124 !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stFileUploader"] button {
    background: #ffffff !important;
    color: #202124 !important;
    border: 1px solid #d5dbe5 !important;
    box-shadow: none !important;
}

div[data-testid="stFileUploader"] button:focus,
div[data-testid="stFileUploader"] button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}

/* Main button */
.stButton > button {
    background: #2457d6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 7px !important;
    box-shadow: none !important;
}

.stButton > button:hover {
    background: #1745b3 !important;
    color: #ffffff !important;
}

/* Candidate names */
.candidate-name {
    font-size: 21px;
    font-weight: 650;
    color: #172033;
    overflow-wrap: anywhere;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        if filename.endswith(".docx"):
            doc = Document(BytesIO(uploaded_file.getvalue()))
            return "\n".join(
                paragraph.text for paragraph in doc.paragraphs
            )

        if filename.endswith(".txt"):
            return uploaded_file.getvalue().decode(
                "utf-8", errors="ignore"
            )

    except Exception as error:
        st.warning(f"Unable to read {uploaded_file.name}: {error}")

    return ""


def extract_candidate_name(text, filename):
    ignored = {
        "resume", "curriculum vitae", "cv", "profile",
        "professional summary", "contact"
    }

    for line in text.splitlines()[:8]:
        line = re.sub(r"\s+", " ", line).strip()
        cleaned = re.sub(r"[^A-Za-z .'-]", "", line).strip()
        words = cleaned.split()

        if (
            2 <= len(words) <= 4
            and cleaned.lower() not in ignored
            and len(cleaned) >= 4
        ):
            return cleaned

    return Path(filename).stem.replace("_", " ").replace("-", " ")


SKILL_LIST = [
    "Python", "Java", "C++", "SQL", "MySQL", "PostgreSQL",
    "Machine Learning", "Deep Learning", "Data Science",
    "Data Analysis", "Statistics", "Pandas", "NumPy",
    "Scikit-learn", "TensorFlow", "PyTorch", "NLP",
    "Generative AI", "RAG", "LangChain", "LangGraph",
    "Power BI", "Tableau", "Excel", "AWS", "Azure",
    "Docker", "Git", "GitHub", "FastAPI", "Flask",
    "Django", "HTML", "CSS", "JavaScript", "React",
    "Node.js", "MongoDB", "Communication", "Leadership",
    "Problem Solving", "Data Visualization", "ETL",
    "Spark", "Hadoop", "Linux", "REST API"
]


def find_skills(text):
    found = []
    text_lower = text.lower()

    for skill in SKILL_LIST:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.append(skill)

    return found


def calculate_candidate(
    text,
    name,
    filename,
    job_description,
    job_skills,
    model,
    vectorizer,
    pdf_bytes=None
):
    if not text or not text.strip():
        return None

    job_vector = vectorizer.transform([job_description])
    resume_vector = vectorizer.transform([text])

    similarity = float(
        cosine_similarity(job_vector, resume_vector)[0][0]
    ) * 100

    resume_skills = find_skills(text)
    resume_skill_set = {skill.lower() for skill in resume_skills}

    matching_skills = [
        skill for skill in job_skills
        if skill.lower() in resume_skill_set
    ]

    missing_skills = [
        skill for skill in job_skills
        if skill.lower() not in resume_skill_set
    ]

    if job_skills:
        coverage = len(matching_skills) / len(job_skills) * 100
    else:
        coverage = similarity

    score = 0.60 * similarity + 0.40 * coverage

    try:
        predicted_category = str(
            model.predict(resume_vector)[0]
        )
    except Exception:
        predicted_category = "Not available"

    return {
        "Candidate Name": name,
        "Resume File": filename,
        "Ranking Score (%)": round(score, 2),
        "Text Similarity (%)": round(similarity, 2),
        "Skill Coverage (%)": round(coverage, 2),
        "Predicted Category": predicted_category,
        "Matching Skills": matching_skills,
        "Skills to Explore": missing_skills,
        "PDF Bytes": pdf_bytes
    }


st.title("ResumeMatch AI")
st.write(
    "Find the best-matching candidates by comparing their resumes "
    "with your job description."
)

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error(
        "Model files not found. Run train_model.py first. "
        "Keep resume_model.pkl and tfidf_vectorizer.pkl "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model, vectorizer = load_models()
except Exception as error:
    st.error(f"Error loading model files: {error}")
    st.stop()


st.subheader("1. Job Description")

job_description = st.text_area(
    "Enter the job description",
    height=170,
    placeholder=(
        "Paste the job responsibilities, required skills, "
        "qualifications and experience..."
    )
)

st.subheader("2. Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload multiple resumes or an Excel file",
    type=["pdf", "docx", "txt", "xlsx", "xls"],
    accept_multiple_files=True
)

st.caption(
    "Excel files should contain a Name column and a Resume Text column. "
    "PDF files retain their original PDF for downloading."
)

if st.button("Find Top 5 Candidates", use_container_width=True):

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    job_skills = find_skills(job_description)
    candidates = []
    file_errors = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        lower_name = filename.lower()

        if lower_name.endswith((".pdf", ".docx", ".txt")):
            text = extract_text(uploaded_file)

            if not text.strip():
                file_errors.append(
                    f"{filename}: no readable text was extracted."
                )
                continue

            pdf_bytes = (
                uploaded_file.getvalue()
                if lower_name.endswith(".pdf")
                else None
            )

            candidates.append({
                "name": extract_candidate_name(text, filename),
                "filename": filename,
                "text": text,
                "pdf_bytes": pdf_bytes
            })

        elif lower_name.endswith((".xlsx", ".xls")):
            try:
                sheets = pd.read_excel(
                    BytesIO(uploaded_file.getvalue()),
                    sheet_name=None
                )

                for sheet_name, df in sheets.items():
                    df.columns = [
                        str(column).strip() for column in df.columns
                    ]

                    name_column = next(
                        (
                            column for column in df.columns
                            if column.lower() in [
                                "name", "candidate name",
                                "candidate", "full name"
                            ]
                        ),
                        None
                    )

                    text_column = next(
                        (
                            column for column in df.columns
                            if column.lower() in [
                                "resume text", "resume",
                                "cv text", "text", "resume content"
                            ]
                        ),
                        None
                    )

                    if text_column is None:
                        file_errors.append(
                            f"{filename}, sheet {sheet_name}: "
                            "missing Resume Text column."
                        )
                        continue

                    for row_index, row in df.iterrows():
                        text = str(row[text_column]).strip()

                        if not text or text.lower() == "nan":
                            continue

                        if name_column and pd.notna(row[name_column]):
                            name = str(row[name_column]).strip()
                        else:
                            name = f"Candidate {row_index + 1}"

                        candidates.append({
                            "name": name,
                            "filename": (
                                f"{filename} - {sheet_name}, "
                                f"row {row_index + 2}"
                            ),
                            "text": text,
                            "pdf_bytes": None
                        })

            except Exception as error:
                file_errors.append(f"{filename}: {error}")

    if file_errors:
        with st.expander("Files needing attention"):
            for message in file_errors:
                st.write(message)

    if not candidates:
        st.error("No readable resumes were found.")
        st.stop()

    results = []
    progress = st.progress(0)

    for index, candidate in enumerate(candidates):
        result = calculate_candidate(
            candidate["text"],
            candidate["name"],
            candidate["filename"],
            job_description,
            job_skills,
            model,
            vectorizer,
            candidate["pdf_bytes"]
        )

        if result is not None:
            results.append(result)

        progress.progress((index + 1) / len(candidates))

    results.sort(
        key=lambda item: item["Ranking Score (%)"],
        reverse=True
    )

    for rank, result in enumerate(results, start=1):
        result["Rank"] = rank

    st.session_state["ranked_candidates"] = results

    st.success(f"Successfully processed {len(results)} resumes.")


if "ranked_candidates" in st.session_state:
    results = st.session_state["ranked_candidates"]
    top_candidates = results[:5]

    st.divider()
    st.subheader("Top 5 Best-Fit Candidates")

    st.caption(
        "Scores are estimated matching scores, not hiring probabilities. "
        "Review each candidate's resume before making a decision."
    )

    for candidate in top_candidates:
        with st.container(border=True):
            left, right = st.columns([2, 1], gap="large")

            with left:
                st.markdown(
                    f'<div class="candidate-name">'
                    f'#{candidate["Rank"]} '
                    f'{html.escape(candidate["Candidate Name"])}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(candidate["Resume File"])

                st.metric(
                    "Match Score",
                    f'{candidate["Ranking Score (%)"]:.1f}%'
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Text Similarity",
                        f'{candidate["Text Similarity (%)"]:.1f}%'
                    )

                with col2:
                    st.metric(
                        "Skill Coverage",
                        f'{candidate["Skill Coverage (%)"]:.1f}%'
                    )

                st.write("**Predicted Resume Category**")
                st.write(candidate["Predicted Category"])

                st.write("**Matching Skills**")
                st.write(
                    ", ".join(candidate["Matching Skills"])
                    if candidate["Matching Skills"]
                    else "No listed skills matched."
                )

                st.write("**Skills to Explore**")
                st.write(
                    ", ".join(candidate["Skills to Explore"])
                    if candidate["Skills to Explore"]
                    else "All detected job-description skills matched."
                )

            with right:
                st.write("**Original Resume PDF**")

                if candidate["PDF Bytes"]:
                    st.download_button(
                        "Download Original PDF",
                        data=candidate["PDF Bytes"],
                        file_name=Path(
                            candidate["Resume File"]
                        ).name,
                        mime="application/pdf",
                        key=(
                            f'download_{candidate["Rank"]}_'
                            f'{candidate["Resume File"]}'
                        )
                    )
                else:
                    st.caption(
                        "No original PDF is available for this candidate."
                    )

    export_rows = []

    for candidate in results:
        export_rows.append({
            "Rank": candidate["Rank"],
            "Candidate Name": candidate["Candidate Name"],
            "Resume File": candidate["Resume File"],
            "Ranking Score (%)": candidate["Ranking Score (%)"],
            "Text Similarity (%)": candidate["Text Similarity (%)"],
            "Skill Coverage (%)": candidate["Skill Coverage (%)"],
            "Predicted Category": candidate["Predicted Category"],
            "Matching Skills": ", ".join(candidate["Matching Skills"]),
            "Skills to Explore": ", ".join(
                candidate["Skills to Explore"]
            )
        })

    export_df = pd.DataFrame(export_rows)

    st.divider()
    st.subheader("Download Results")

    st.download_button(
        "Download All Candidate Rankings (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="resume_match_results.csv",
        mime="text/csv"
    )
