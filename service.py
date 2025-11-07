import os
from typing import Literal, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb


load_dotenv()

_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        _client = OpenAI(api_key=api_key)
    return _client


_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
_embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

_chroma_host = os.getenv("CHROMA_HOST", "localhost")
_chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
_collection_name = os.getenv("CHROMA_COLLECTION", "rag_docs")

_chroma_client = chromadb.HttpClient(host=_chroma_host, port=_chroma_port)
_vectorstore = Chroma(
    client=_chroma_client,
    collection_name=_collection_name,
    embedding_function=_embedding_model,
)

try:
    if _chroma_client.get_or_create_collection(_collection_name).count() == 0:
        pdf_path = os.getenv("RAG_PDF_PATH", "SriLanka_Tourism.pdf")
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            _vectorstore.add_documents(splits)
except Exception:
    pass

_retriever = _vectorstore.as_retriever()

_system_prompt = (
    "You are an intelligent chatbot. Use the following context to answer the question. "
    "If you don't know the answer, just say that you don't know.\n\n{context}"
)
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system_prompt),
        ("human", "{input}"),
    ]
)
_qa_chain = create_stuff_documents_chain(_llm, _prompt)
rag_chain = create_retrieval_chain(_retriever, _qa_chain)


def answer_with_rag(question: str) -> str:
    result = rag_chain.invoke({"input": question})
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]
    return str(result)

_pg_user = os.getenv("POSTGRES_USER")
_pg_password = os.getenv("POSTGRES_PASSWORD")
_pg_host = os.getenv("POSTGRES_HOST")
_pg_port = int(os.getenv("POSTGRES_PORT") or 5432)
_pg_db = os.getenv("POSTGRES_DB")

_engine = create_engine(
    f"postgresql+psycopg2://{_pg_user}:{_pg_password}@{_pg_host}:{_pg_port}/{_pg_db}",
    pool_pre_ping=True,
)

_few_shot_examples = """
Q: List vehicles with their owner name and contact.
SQL: SELECT v.vehicle_id, v.vin, v.license_plate, v.model, v.make, v.year,
            c.name AS customer_name, c.email, c.phone
     FROM vehicle_service_db.vehicle AS v
     JOIN customer_service_db.customer AS c ON v.customer_id = c.customer_id;

Q: Show upcoming appointments with customer and vehicle details.
SQL: SELECT a.appointment_id, a.appointment_date, a.time_slot, a.service_type, a.status,
            c.name AS customer_name, v.license_plate, v.model, v.make
     FROM appointment_service_db.appointment AS a
     JOIN customer_service_db.customer AS c ON a.customer_id = c.customer_id
     JOIN vehicle_service_db.vehicle AS v ON a.vehicle_id = v.vehicle_id
     ORDER BY a.appointment_date ASC;

Q: List services with assigned employee and vehicle info.
SQL: SELECT s.service_id, s.service_type, s.status, s.start_time, s.end_time,
            e.name AS employee_name, v.license_plate, v.model
     FROM service_management_db.service AS s
     JOIN employee_service_db.employee AS e ON s.assigned_to = e.employee_id
     JOIN vehicle_service_db.vehicle AS v ON s.vehicle_id = v.vehicle_id;

Q: Show updates for service 1 ordered by time.
SQL: SELECT u.update_id, u.progress_percentage, u.update_text, u.created_at
     FROM service_management_db.service_update AS u
     WHERE u.service_id = 1
     ORDER BY u.created_at ASC;

Q: Total time logged per employee.
SQL: SELECT e.employee_id, e.name, SUM(t.duration_minutes) AS total_minutes
     FROM employee_service_db.employee AS e
     JOIN employee_service_db.time_log AS t ON e.employee_id = t.employee_id
     GROUP BY e.employee_id, e.name
     ORDER BY total_minutes DESC;
"""


def generate_sql(question: str) -> str:
    prompt = f"""
You are an expert PostgreSQL SQL generator for a garage service microservices database.

Schemas/tables:
- customer_service_db.customer (customer_id, first_name, last_name, name, email, phone, password_hash, created_at)
- employee_service_db.employee (employee_id, first_name, last_name, name, email, password_hash, role, photo_url, status, hourly_rate, specialization, hire_date, created_at)
- employee_service_db.time_log (log_id, employee_id, work_type, start_time, end_time, duration_minutes, description, created_at)
- vehicle_service_db.vehicle (vehicle_id, customer_id, vin, license_plate, model, make, year, color, mileage, status, updated_at)
- appointment_service_db.appointment (appointment_id, customer_id, vehicle_id, appointment_date, time_slot, service_type, status, created_at, updated_at)
- service_management_db.service (service_id, vehicle_id, assigned_to, service_type, description, start_time, end_time, estimated_cost, actual_cost, completion_percentage, notes, status, created_at, updated_at)
- service_management_db.service_update (update_id, service_id, progress_percentage, update_text, created_at)

Relationships:
- vehicle.customer_id → customer.customer_id
- appointment.customer_id → customer.customer_id; appointment.vehicle_id → vehicle.vehicle_id
- service.vehicle_id → vehicle.vehicle_id; service.assigned_to → employee.employee_id
- service_update.service_id → service.service_id

Requirements:
- ALWAYS fully-qualify tables with schema (e.g., service_management_db.service).
- Generate ONE valid SELECT statement only. No comments, no markdown, no DDL/DML.
- Prefer JOINs to combine related entities according to the relationships above.
- Include ORDER BY when the question implies sorting (e.g., latest, top, upcoming).

Examples:
{_few_shot_examples}

Q: {question}
SQL:
"""
    response = _get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    sql_query = response.choices[0].message.content.strip()
    return sql_query.replace("```sql", "").replace("```", "").strip()


def execute_sql(query: str):
    try:
        with _engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        return str(e)


def explain_results(question: str, df):
    if isinstance(df, str):
        return f"I couldn’t process your question because of an error: {df}"

    if df.empty:
        return "I couldn’t find any matching records for that question."

    data_str = df.to_string(index=False)

    prompt = f"""
You are a friendly garage service assistant.
The user asked: "{question}"

Here are the query results:
{data_str}

Write a short and natural explanation in plain language — no SQL, just summarize like you’re talking to the user.
"""
    response = _get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def answer_with_sql(question: str) -> str:
    sql_query = generate_sql(question)
    df_or_err = execute_sql(sql_query)
    reply = explain_results(question, df_or_err)
    return reply

def classify_question(question: str) -> Literal["rag", "sql"]:
    """
    Decide whether to answer a question using RAG (document retrieval) or SQL (database query).

    Use SQL when the user asks for database-backed facts, records, lists, counts, aggregations,
    filtering by columns (date, status, id, VIN, license plate), joins across entities
    (customer/vehicle/appointment/service/employee/time logs), or schema/column/table details.
    Examples: "How many appointments this week?", "Show services for vehicle ABC-123",
    "List employees with total time logged", "Latest updates for service 12".

    Use RAG for brochure/document-style knowledge, explanations, procedures, policies, FAQs, and
    general informational questions grounded in the document store/PDFs (e.g., brochure/prospectus).
    Examples: "What services do you offer?", "Explain the booking process", "Give an overview".
    """

    client = _get_openai_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "use_rag",
                "description": (
                    "Choose this for brochure/document-style knowledge: overviews, explanations,"
                    " procedures, policies, FAQs, or content grounded in PDFs/docs (brochure/"
                    "prosure/prospectus)."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "use_sql",
                "description": (
                    "Choose this for database questions: fetching rows, lists, details, counts,"
                    " sums/averages, filters by date/status/id/VIN/license plate, joins across"
                    " customer/vehicle/appointment/service/employee/time logs, or schema questions."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]

    system = (
        "You are a strict router. Read the user question and call exactly one tool.\n\n"
        "Call 'use_sql' IF AND ONLY IF the question requires database-backed answers such as:"
        " rows/records, lists, details for a specific entity (by id/email/VIN/license plate),"
        " counts/totals/averages, trends, filters by date/status, top/latest/upcoming, joins"
        " between customers/vehicles/appointments/services/employees/time logs, or schema/column"
        " information.\n"
        "Examples → SQL: 'How many appointments this week?', 'Show services for ABC-123',"
        " 'List employees with total time logged', 'Latest updates for service 12'.\n\n"
        "Otherwise, call 'use_rag' for brochure/document knowledge: overviews, explanations,"
        " procedures, policies, FAQs, and general info grounded in PDFs/docs (brochure/prospectus).\n"
        "Examples → RAG: 'What services do you offer?', 'Explain booking process', 'Give an overview'.\n"
        "Do not output text; always call exactly one tool."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        tools=tools,
        tool_choice="auto",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        lowered = question.lower()

        sql_keywords = [
            "sql", "query", "postgres", "database", "table", "column", "schema",
            "count", "total", "sum", "average", "avg", "min", "max",
            "trend", "top", "latest", "upcoming", "recent", "overdue",
            "where", "between", "before", "after", "since", "today", "yesterday",
            "this week", "this month", "last month", "order by", "group by",
            "list", "show", "get", "find", "fetch", "display",
            "customer", "employee", "vehicle", "appointment", "service", "time_log", "service_update",
            "id", "vin", "license_plate", "model", "make", "year", "mileage", "status",
            "estimated_cost", "actual_cost", "hire_date", "progress_percentage", "completion_percentage",
        ]

        rag_keywords = [
            "what is", "who is", "how to", "how do i", "explain", "overview", "about",
            "benefits", "features", "pros and cons", "steps", "guide", "manual",
            "policy", "procedure", "brochure", "prosure", "prospectus", "document", "pdf",
            "faq", "terms", "conditions", "opening hours", "contact", "pricing"
        ]

        sql_hit = any(k in lowered for k in sql_keywords)
        rag_hit = any(k in lowered for k in rag_keywords)

        if sql_hit and (not rag_hit or any(e in lowered for e in [
            "customer", "employee", "vehicle", "appointment", "service", "time_log", "service_update",
            "count", "total", "sum", "average", "avg", "latest", "upcoming"
        ])):
            return "sql"

        if rag_hit:
            return "rag"

        if any(w in lowered for w in ["how many", "number of", "total", "count"]):
            return "sql"
        return "rag"

    fn = tool_calls[0].function.name
    return "sql" if fn == "use_sql" else "rag"


def answer_question(question: str) -> dict:
    route: Literal["rag", "sql"] = classify_question(question)
    if route == "sql":
        answer = answer_with_sql(question)
    else:
        answer = answer_with_rag(question)
    return {"route": route, "answer": answer}


