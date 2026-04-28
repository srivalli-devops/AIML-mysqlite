import streamlit as st
import pandas as pd
import re
from pathlib import Path
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# --- Page config ---
st.set_page_config(page_title="Fast SQL Chat", page_icon="⚡")
st.title("⚡ Fast Chat with SQL")

# --- Sidebar ---
radio_opt = ["Use SQLite (imdb.db)", "Connect to MySQL"]
selected_opt = st.sidebar.radio("Choose database", options=radio_opt)

LOCALDB = "LOCALDB"
MYSQL = "MYSQL"

if selected_opt == radio_opt[0]:
    db_mode = LOCALDB
else:
    db_mode = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", value="127.0.0.1")
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database", value="project")

api_key = st.sidebar.text_input("Groq API Key", type="password")

if not api_key:
    st.info("Enter your Groq API key to continue.")
    st.stop()

# --- LLM ---
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    streaming=False
)

# --- Prompt ---
prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template="""
You are a SQL expert.

STRICT RULES:
- Use ONLY the tables and columns provided
- DO NOT invent tables
- DO NOT use unnecessary JOINs
- Prefer single-table queries when possible
- Return ONLY SQL

Schema:
{table_info}

Limit results to {top_k}

Question: {input}
SQL:
"""
)

# --- Clean SQL ---
def clean_sql(query: str) -> str:
    query = query.strip()
    if query.startswith("```"):
        query = query.replace("```sql", "").replace("```", "").strip()
    if query.lower().startswith("sql"):
        query = query[3:].strip()
    for tag in ["SQLQuery:", "SQLResult:", "Answer:"]:
        if tag in query:
            query = query.split(tag)[-1].strip()
    return query

# --- Guardrails ---
def get_schema_info(db):
    return db.get_table_info()

def validate_sql(sql, schema_info):
    errors = []
    sql_lower = sql.lower()

    tables = re.findall(r'create table (\w+)', schema_info.lower())

    matches = re.findall(r'from\s+(\w+)|join\s+(\w+)', sql_lower)
    for match in matches:
        table = next(filter(None, match))
        if table not in tables:
            errors.append(f"Invalid table: {table}")

    if " join " in sql_lower and " on " not in sql_lower:
        errors.append("JOIN without ON")

    if any(x in sql_lower for x in ["drop", "delete", "truncate", "update", "insert"]):
        errors.append("Dangerous query")

    return errors

def fix_sql(chain, bad_sql, errors):
    feedback = f"""
Fix this SQL:

{bad_sql}

Errors:
{errors}

Return ONLY corrected SQL.
"""
    return clean_sql(chain.invoke({"question": feedback}))

def safe_sql_execution(chain, db, user_prompt, max_retries=2):
    schema = get_schema_info(db)

    sql_query = clean_sql(chain.invoke({"question": user_prompt}))

    for _ in range(max_retries):
        errors = validate_sql(sql_query, schema)

        if not errors:
            try:
                engine = db._engine
                with engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    rows = result.fetchall()
                    columns = result.keys()
                return rows, columns, sql_query
            except Exception as e:
                errors.append(str(e))

        sql_query = fix_sql(chain, sql_query, errors)

    raise Exception(f"Failed query: {sql_query}")

# --- DB ---
@st.cache_resource
def get_db():
    if db_mode == LOCALDB:
        db_file = Path(__file__).parent / "imdb.db"
        engine = create_engine(f"sqlite:///{db_file}")
    else:
        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        )
    return SQLDatabase(engine)

@st.cache_resource
def get_chain(_db):
    return create_sql_query_chain(llm, _db, prompt=prompt)

db = get_db()
chain = get_chain(db)

# --- Chat history ---
if "messages" not in st.session_state or st.sidebar.button("Clear history"):
    st.session_state.messages = [
        {"role": "assistant", "type": "text", "content": "Ask me anything about your database."}
    ]

# --- Display messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.write(msg["content"])
        elif msg["type"] == "table":
            st.dataframe(msg["content"], use_container_width=True)

# --- Input ---
user_prompt = st.chat_input("Ask a question...")

if user_prompt:
    st.session_state.messages.append({
        "role": "user",
        "type": "text",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):

                rows, columns, sql_query = safe_sql_execution(chain, db, user_prompt)

                df = pd.DataFrame(rows, columns=columns)

                if df.empty:
                    msg = "I couldn’t find any matching records."
                    st.write(msg)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "text",
                        "content": msg
                    })
                else:
                    df = df.head(100)
                    msg = f"I found {len(df)} results:"
                    st.write(msg)
                    st.dataframe(df, use_container_width=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "text",
                        "content": msg
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "table",
                        "content": df
                    })

                with st.expander("🔍 SQL Debug"):
                    st.code(sql_query, language="sql")

        except Exception as e:
            err = f"❌ Error: {e}"
            st.write(err)
            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": err
            })
