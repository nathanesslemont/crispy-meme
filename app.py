"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated route only) -> escalate to human OR
    execute_query -> generate_response

Run locally:
    streamlit run app.py
"""

import os
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# File locations (override with environment variables if needed)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("DB_PATH", str(APP_DIR / "credit_risk_portfolio.db"))
GROUND_TRUTH_PATH = os.environ.get("GROUND_TRUTH_PATH", str(APP_DIR / "test_queries.csv"))
CONFIG_PATH = os.environ.get("CONFIG_PATH", str(APP_DIR / "config.json"))
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", str(APP_DIR / "audit_log.jsonl"))
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# OpenAI credentials: Streamlit secrets -> environment variables -> config.json
# ---------------------------------------------------------------------------
def load_credentials():
    """Return (api_key, api_base). Same two values the notebook read from config.json."""
    api_key, api_base = None, None

    # 1) Streamlit secrets (Streamlit Community Cloud / .streamlit/secrets.toml)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE") or st.secrets.get("OPENAI_BASE_URL")
    except Exception:
        pass

    # 2) Environment variables
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    # 3) config.json (same format as the notebook)
    if (not api_key) and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as file:
            config = json.load(file)
            api_key = config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if not OPENAI_API_KEY:
    st.error(
        "OpenAI credentials not found. Provide `OPENAI_API_KEY` (and optionally "
        "`OPENAI_API_BASE`) via Streamlit secrets, environment variables, or a `config.json` file."
    )
    st.stop()

# Store API credentials in environment variables (as in the notebook)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


# ---------------------------------------------------------------------------
# LLM initialisation (same models/temperature as the notebook)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_llms(model_name):
    llm_ = ChatOpenAI(model=model_name, temperature=0.0)
    evaluator_llm_ = ChatOpenAI(model=model_name, temperature=0.0)
    return llm_, evaluator_llm_


llm, evaluator_llm = init_llms(MODEL_NAME)


# ---------------------------------------------------------------------------
# Database loading (read-only connection)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    """Connect to the SQLite database in read-only mode (?mode=ro) via URI."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    # check_same_thread=False: Streamlit executes script reruns on different threads.
    # The connection is read-only, so sharing it is safe.
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


if not os.path.exists(DB_PATH):
    st.error(
        f"Database file not found at `{DB_PATH}`. Place `credit_risk_portfolio.db` next to "
        "`app.py` or set the `DB_PATH` environment variable."
    )
    st.stop()

conn = get_connection(DB_PATH)

# ---------------------------------------------------------------------------
# Database schema provided to the LLM in prompts (verbatim from the notebook)
# ---------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ---------------------------------------------------------------------------
# CSV loading (ground truth test cases) - optional for the Q&A tab
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_ground_truth(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

# ---------------------------------------------------------------------------
# Verified Query Template Library (SQL verbatim from the notebook)
# ---------------------------------------------------------------------------
sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_millions,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_millions
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_millions DESC
"""

sql_2 = """
SELECT
    loan_category,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions,
    COUNT(loan_account_number) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_millions DESC
"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""

sql_4 = """
SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS average_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY average_provision_coverage_ratio DESC
"""

sql_5 = """
SELECT
    borrower_name,
    sector_code,
    ROUND(total_outstanding / 1000000.0, 2) AS total_outstanding_millions,
    asset_classification
FROM loan_master
ORDER BY total_outstanding DESC
LIMIT 10
"""

sql_6 = """
SELECT
    group_name,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL AND group_name != 'None'
GROUP BY group_name
ORDER BY total_outstanding_millions DESC
LIMIT 5
"""

sql_7 = """
SELECT
    loan_account_number,
    borrower_name,
    sector_code,
    ROUND(total_outstanding / 1000000.0, 2) AS total_outstanding_millions,
    days_past_due,
    asset_classification
FROM loan_master
WHERE days_past_due > 0
ORDER BY days_past_due DESC
"""

sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        WHEN days_past_due > 90 THEN '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END
"""

sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ===========================================================================
# TOOL DEFINITIONS (logic preserved from the notebook)
# ===========================================================================

# ---------------------------------------------------------------------------
# Intent Classification Tool
# ---------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.
    '''
    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
Analyze the following user question and decide if it matches one of the pre-approved queries in the verified library, or if it requires a custom generated query.

User Question:
{user_question}

Verified Library:
{library_descriptions}

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ---------------------------------------------------------------------------
# Query Generation Tool
# ---------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.
    '''
    generation_prompt = f"""
Using the SQLite database schema provided below, generate a read-only SELECT query to answer the user's question. Use correct joins and aggregate calculations where required.

User Question:
{user_question}

Database Schema:
{schema_context}

Your output must be the raw, valid SQL query only. Do not enclose it in markdown formatting or add any narrative.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ---------------------------------------------------------------------------
# Query Validation Tool (five checks)
# ---------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.
    '''
    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
Evaluate whether the candidate SQL correctly addresses the user's query intent, using appropriate database logic.

Context:
{track_context}

User Question:
{user_question}

Candidate SQL:
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ---------------------------------------------------------------------------
# Retry Generation Tool
# ---------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.
    '''
    retry_prompt = f"""
Your previous SQL query failed validation checks. Please rewrite the query to fix the error while keeping the original question intent.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

Provide only the raw SQL query in your response, without any markdown formatting.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ---------------------------------------------------------------------------
# Query Execution Tool
# ---------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ---------------------------------------------------------------------------
# Response Generation Tool
# ---------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.
    '''
    response_prompt = f"""
Answer the user's commercial credit portfolio question based on the query result provided below.

User Question:
{user_question}

Query Results (Pandas DataFrame):
{dataframe.to_string()}

Write a brief, professional summary of the results answering exactly what was asked. Focus on the core numbers.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# (print() calls from the notebook are collected into a `trace` list so the UI
#  can show them in an expander instead of the console)
# ---------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages in `trace`.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, log and trace.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }
    trace = []

    def say(message):
        if verbose:
            trace.append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, 'trace': trace, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, 'trace': trace, 'warnings': exec_result['warnings'], **log}


# ===========================================================================
# AUDIT TRAIL
# ===========================================================================
def build_audit_entry(result):
    """Flatten a pipeline result into one auditable record."""
    gate = result.get('gate_result') or {}
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'user_question': result.get('user_question'),
        'route': result.get('route'),
        'query_id': result.get('query_id'),
        'match_reason': result.get('match_reason'),
        'retry_used': result.get('retry_used'),
        'escalated': result.get('escalated'),
        'gate_passed': gate.get('passed'),
        'gate_failed_check': gate.get('failed_check'),
        'gate_details': gate.get('details'),
        'confidence': result.get('confidence'),
        'row_count': result.get('row_count'),
        'sql': result.get('executed_sql') or result.get('candidate_sql'),
        'narrative': result.get('narrative'),
    }


def persist_audit_entry(entry):
    """Append the record to a JSON-lines file (best effort)."""
    try:
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, default=str) + '\n')
    except Exception:
        pass


def run_and_record(question):
    """Run the pipeline, store the result for rendering and append to the audit trail."""
    result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
    entry = build_audit_entry(result)
    st.session_state.setdefault('audit_log', []).append(entry)
    persist_audit_entry(entry)
    return result


# ===========================================================================
# STREAMLIT UI
# ===========================================================================
def render_result(res):
    """Display the answer, confidence, SQL used, raw data and validation details."""
    route = res['route']
    query_id = res.get('query_id')

    # --- Route banner -------------------------------------------------------
    if res.get('escalated'):
        st.error("🚨 Escalated to a human analyst — the query could not be reliably resolved.")
    elif route == 'verified':
        st.success(f"✅ Verified template used: **{query_id}** — {verified_query_library[query_id]['description']}")
    else:
        st.info("🧩 No verified template matched — SQL was generated for this question (read-only SELECT).")

    # --- Confidence + metadata ---------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    conf = res.get('confidence')
    if isinstance(conf, (int, float)):
        c1.metric("Confidence", f"{conf:.2f}")
    else:
        c1.metric("Confidence", str(conf))
    c2.metric("Route", str(route).capitalize())
    c3.metric("Query ID", query_id if query_id else "—")
    c4.metric("Rows returned", res.get('row_count') if res.get('row_count') is not None else "—")

    if isinstance(conf, (int, float)):
        st.progress(min(max(float(conf), 0.0), 1.0))

    st.caption(f"Routing reason: {res.get('match_reason')}")

    # --- Answer -------------------------------------------------------------
    st.subheader("Answer")
    if res.get('escalated'):
        st.warning(res['narrative'])
    else:
        st.write(res['narrative'])

    for w in res.get('warnings', []) or []:
        st.warning(f"Data check: {w}")

    # --- SQL used -----------------------------------------------------------
    st.subheader("SQL Query Used")
    if res.get('retry_used'):
        st.caption("⚠️ The first generated query failed validation; this is the retried query.")
    shown_sql = res.get('executed_sql') or res.get('candidate_sql')
    if res.get('escalated'):
        st.caption("This SQL was NOT executed (validation failed). Shown for analyst review.")
    st.code(shown_sql or "", language="sql")

    # --- Raw data -----------------------------------------------------------
    st.subheader("Raw Data Returned")
    df = res.get('dataframe')
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download result as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"query_result_{query_id or 'generated'}.csv",
            mime="text/csv",
        )
    else:
        st.write("No data returned (query escalated).")

    # --- Validation gate + trace -------------------------------------------
    with st.expander("Validation gate details"):
        gate = res.get('gate_result') or {}
        st.json({
            'passed': gate.get('passed'),
            'failed_check': gate.get('failed_check'),
            'details': gate.get('details'),
            'relevance_confidence': gate.get('relevance_confidence'),
            'retry_used': res.get('retry_used'),
            'escalated': res.get('escalated'),
        })

    with st.expander("Pipeline trace"):
        st.code("\n".join(res.get('trace', [])) or "(no trace)", language="text")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
st.caption(
    "Ask routine commercial lending portfolio questions in plain English. "
    "Answers come from pre-approved verified SQL templates or validated, read-only generated SQL. "
    "Every answer shows the SQL used, the raw data and a confidence score."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Scope:** commercial lending portfolio, read-only analytics\n"
        "- **Database:** SQLite, opened in read-only mode\n"
        f"- **Model:** `{MODEL_NAME}` (temperature 0)\n"
        "- **Fallback:** failed generated queries are retried once, then escalated to a human analyst"
    )
    st.divider()
    st.subheader("Verified Query Library")
    for qid, entry in verified_query_library.items():
        st.markdown(f"**{qid}** — {entry['description']}")
    st.divider()
    with st.expander("Database schema given to the LLM"):
        st.code(database_schema, language="text")
    with st.expander("View verified SQL templates"):
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}**")
            st.code(entry['sql'].strip(), language="sql")

tab_ask, tab_eval, tab_audit = st.tabs(["💬 Ask a Question", "🧪 Evaluation (Ground Truth)", "📜 Audit Trail"])

# ---------------------------------------------------------------------------
# Tab 1: Ask a question
# ---------------------------------------------------------------------------
with tab_ask:
    if 'question_text' not in st.session_state:
        st.session_state['question_text'] = ""

    def _load_example():
        choice = st.session_state.get('example_select')
        if choice and choice != "— choose an example —":
            st.session_state['question_text'] = choice

    if ground_truth is not None and 'User Query' in ground_truth.columns:
        st.selectbox(
            "Sample questions (from test_queries.csv)",
            ["— choose an example —"] + ground_truth['User Query'].tolist(),
            key="example_select",
            on_change=_load_example,
        )

    st.text_area(
        "Your question",
        key="question_text",
        height=100,
        placeholder="e.g., What is the DPD aging profile of the portfolio?",
    )

    if st.button("Run query", type="primary"):
        question = st.session_state['question_text'].strip()
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running pipeline: classify → build SQL → validate → execute → summarise..."):
                try:
                    st.session_state['last_result'] = run_and_record(question)
                    st.session_state['last_error'] = None
                except Exception as e:
                    st.session_state['last_result'] = None
                    st.session_state['last_error'] = f"{type(e).__name__}: {e}"

    if st.session_state.get('last_error'):
        st.error(f"The pipeline failed unexpectedly. Please retry or escalate to an analyst.\n\n{st.session_state['last_error']}")

    if st.session_state.get('last_result'):
        render_result(st.session_state['last_result'])

# ---------------------------------------------------------------------------
# Tab 2: Evaluation against ground truth
# ---------------------------------------------------------------------------
with tab_eval:
    st.markdown(
        "Runs every test case in `test_queries.csv` through the pipeline and compares the actual route and "
        "query ID with the expected ones. This calls the LLM several times per test case."
    )

    if ground_truth is None:
        st.warning(f"`test_queries.csv` not found at `{GROUND_TRUTH_PATH}`. Add it next to `app.py` to enable evaluation.")
    else:
        st.dataframe(ground_truth, use_container_width=True)

        if st.button("Run evaluation on all test cases"):
            test_results = []
            progress = st.progress(0.0)
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                with st.spinner(f"Running {gt['Test Case']} ({i + 1}/{len(ground_truth)})..."):
                    try:
                        test_results.append(run_and_record(gt['User Query']))
                    except Exception as e:
                        test_results.append({
                            'route': None, 'query_id': None, 'confidence': f"ERROR: {e}", 'row_count': None
                        })
                progress.progress((i + 1) / len(ground_truth))

            evaluation_rows = []
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = test_results[i]

                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['Expected Route'],
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count']
                })

            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df['Route Match'].mean() * 100

            verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
            query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

            # Safely convert confidence values to numeric, turning 'ESCALATED' or other non-numeric strings into NaN
            numeric_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce')
            average_confidence = numeric_confidence.mean()

            st.session_state['evaluation'] = {
                'df': evaluation_df,
                'path_accuracy': path_accuracy,
                'query_accuracy': query_accuracy,
                'average_confidence': average_confidence,
            }

        ev = st.session_state.get('evaluation')
        if ev:
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{ev['path_accuracy']:.1f}%")
            m2.metric(
                "Selected Query Accuracy",
                "N/A" if pd.isna(ev['query_accuracy']) else f"{ev['query_accuracy']:.1f}%",
            )
            m3.metric(
                "Average Confidence Score",
                "N/A" if pd.isna(ev['average_confidence']) else f"{ev['average_confidence']:.2f}",
            )
            st.subheader("Detailed Evaluation Results")
            st.dataframe(ev['df'].astype({'Confidence': str}), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Audit trail
# ---------------------------------------------------------------------------
with tab_audit:
    st.markdown(
        "Every analytical output produced in this session is recorded here with its route, SQL, validation "
        f"outcome and confidence. Records are also appended to `{os.path.basename(AUDIT_LOG_PATH)}` on the "
        "server (note: the file system on Streamlit Community Cloud is not persistent)."
    )
    audit_log = st.session_state.get('audit_log', [])
    if audit_log:
        audit_df = pd.DataFrame(audit_log)
        st.dataframe(audit_df, use_container_width=True)
        st.download_button(
            "⬇️ Download audit trail (CSV)",
            data=audit_df.to_csv(index=False).encode('utf-8'),
            file_name="audit_trail.csv",
            mime="text/csv",
        )
    else:
        st.info("No queries have been run in this session yet.")
