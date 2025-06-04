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

tsx_path = "summary.txt"

# Gene summarization function
def gene_summarizer(id, text):
    prompt = f"As a biological data expert, analyze the following structured gene data and provide a concise, 2-sentence summary. Gene data: {text}"
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)

    summarized_txt = response.text.strip()
    
    with open(tsx_path, 'a') as f:
        f.write(f'--- Gene ID: {id} ---\n')
        f.write(f'Summary: {summarized_txt}\n\n')

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