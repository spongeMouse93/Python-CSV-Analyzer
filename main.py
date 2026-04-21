import os, pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from utils import logger, validate_llm_output

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI KEY"))

def analyze_data_quality(file_path):
    logger.info(f"Starting analysis for {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded {len(df)} rows.")
    summary = {
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "row_count": len(df)
    }
    prompt = f"""
    You are a data quality assistant. 
    Dataset summary: {summary}
    Sample rows: {df.head(5).to_dict()} 

    Tasks:
    1. Identify data quality issues [cite: 57]
    2. Suggest fixes [cite: 58]
    3. Output strictly in JSON:
    {{
        "issues": [],
        "recommendations": [],
        "severity": "low/medium/high"
    }}
    """
    logger.info("Sending summary to LLM for reasoning...")
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{"role": "user", "content": prompt}],
        temperature = 0
    )
    raw_content = response.choices[0].message.content
    is_valid, structured_report = validate_llm_output(raw_content)
    if is_valid:
        logger.info("Analysis complete. Structured report generated.")
        print("\n=== DATA QUALITY REPORT ===")
        print(f"Severity: {structured_report['severity'].upper()}")
        for issue in structured_report['issues']:
            print(f"- {issue}")
    else:
        logger.error("LLM returned invalid JSON. Manual review required.")

if __name__ == "__main__":
    analyze_data_quality("diabetes.csv")
