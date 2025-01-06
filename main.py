import os
from pathlib import Path
from dotenv import load_dotenv
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.model.groq import Groq
import streamlit as st

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
TMP_DIR = BASE_DIR.joinpath("tmp")
TMP_DIR.mkdir(exist_ok=True, parents=True)

CSV_PATH = "IMDB-Movie-Data.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File {CSV_PATH} not found in project directory")

agent = PythonAgent(
    model=Groq(id="llama-3.1-70b-versatile"),
    base_dir=TMP_DIR,
    files=[
        CsvFile(
            path=CSV_PATH,
            description="IMDB movies dataset with genres, directors, and ratings"
        )
    ],
    markdown=True,
    pip_install=True,
    show_tool_calls=True
)

def main():
    st.title("IMDB Movie Data Explorer")
    
    question = st.text_area("Ask about IMDB movies:", placeholder="e.g., Highest rated movie?")
    
    if st.button("Search"):
        if question.strip():
            try:
                with st.spinner("Analyzing..."):
                    st.markdown(agent.run(question).content)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("Please enter a question")

if __name__ == "__main__":
    main()