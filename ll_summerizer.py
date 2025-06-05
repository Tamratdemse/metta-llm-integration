import google.generativeai as genai
from hyperon.atoms import OperationAtom, S
from hyperon.ext import register_atoms
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

# Define both file paths
summary_txt_path = "summary.txt"
summary_metta_path = "summary.metta"

# Gene summarization function
def gene_summarizer(id, text):
    prompt = f"As a biological data expert, analyze the following structured gene data and provide a concise, 2-sentence summary. Gene data: {text}"
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)

    summarized_txt = response.text.strip()
    
    # Write to summary.txt
    with open(summary_txt_path, 'a') as f_txt:
        f_txt.write(f'--- Gene ID: {id} ---\n')
        f_txt.write(f'Summary: {summarized_txt}\n\n')

    # Write to summary.metta
    with open(summary_metta_path, 'a') as f_metta:
        f_metta.write(f'(Summary (gene {id}) "{ summarized_txt}" )\n\n')

    return [S(summarized_txt)]

# Registering atom to Hyperon
@register_atoms(pass_metta=True)
def utils(metta):
    summaryGene = OperationAtom(
        "gene_summarizer",
        lambda id, text: gene_summarizer(id, text),
        ["Atom", "Expression", "Expression"],
        unwrap=False
    )
    return {r"gene_summarizer": summaryGene}