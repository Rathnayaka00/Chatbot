import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI(api_key=api_key)

rag_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_path = os.getenv("CHROMA_PATH", "chroma_db")
collection_name = os.getenv("CHROMA_COLLECTION", "rag_docs")

chroma_client = chromadb.PersistentClient(path=chroma_path)
vectorstore = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embedding_model,
)

if chroma_client.get_or_create_collection(collection_name).count() == 0:
    print("Vector store is empty. Loading 'SriLanka_Tourism.pdf'...")
    try:
        loader = PyPDFLoader("SriLanka_Tourism.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        vectorstore.add_documents(splits)
        print("Documents loaded and vectorized.")
    except Exception as e:
        print(f"Error loading PDF: {e}. Make sure 'SriLanka_Tourism.pdf' is in the correct directory.")

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an intelligent chatbot. Use the following context to answer the question. "
    "If you don't know the answer, just say that you don't know."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(rag_llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

def invoke_rag_chain(question: str) -> str:
    try:
        response = rag_chain.invoke({"input": question})
        return response["answer"]
    except Exception as e:
        return f"Error processing RAG query: {str(e)}"

pg_user = os.getenv("POSTGRES_USER")
pg_password = os.getenv("POSTGRES_PASSWORD")
pg_host = os.getenv("POSTGRES_HOST")
pg_port = int(os.getenv("POSTGRES_PORT") or 5432)
pg_db = os.getenv("POSTGRES_DB")

try:
    engine = create_engine(
        f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}",
        pool_pre_ping=True
    )
    # Test connection
    with engine.connect() as conn:
        print("Database connection successful.")
except Exception as e:
    print(f"Database connection error: {e}")
    engine = None 

few_shot_examples = """
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

Q: I want details about the service.
SQL: SELECT s.service_id,
           s.service_type,
           s.status,
           s.start_time,
           s.end_time,
           s.completion_percentage,
           e.name AS employee_name,
           v.license_plate,
           v.model,
           v.make,
           c.name AS customer_name
      FROM service_management_db.service AS s
      LEFT JOIN employee_service_db.employee AS e ON s.assigned_to = e.employee_id
      LEFT JOIN vehicle_service_db.vehicle AS v ON s.vehicle_id = v.vehicle_id
      LEFT JOIN customer_service_db.customer AS c ON v.customer_id = c.customer_id
      ORDER BY s.start_time DESC NULLS LAST;

Q: Get the latest updates about the service 3.
SQL: SELECT u.update_id,
           u.progress_percentage,
           u.update_text,
           u.created_at
      FROM service_management_db.service_update AS u
      WHERE u.service_id = 3
      ORDER BY u.created_at DESC;
 
Q: What are the available services?
SQL: SELECT DISTINCT s.service_type
      FROM service_management_db.service AS s
      WHERE s.service_type IS NOT NULL
      ORDER BY s.service_type ASC;

Q: Who are our staff members?
SQL: SELECT e.employee_id, e.name, e.role, e.status, e.hire_date
      FROM employee_service_db.employee AS e
      ORDER BY e.name ASC;

Q: Which cars do we have?
SQL: SELECT v.vehicle_id, v.license_plate, v.model, v.make, v.year, v.status,
           c.name AS customer_name
      FROM vehicle_service_db.vehicle AS v
      LEFT JOIN customer_service_db.customer AS c ON v.customer_id = c.customer_id
      ORDER BY v.updated_at DESC NULLS LAST;

Q: Show upcoming bookings.
SQL: SELECT a.appointment_id, a.appointment_date, a.time_slot, a.service_type, a.status,
           c.name AS customer_name, v.license_plate, v.model, v.make
      FROM appointment_service_db.appointment AS a
      LEFT JOIN customer_service_db.customer AS c ON a.customer_id = c.customer_id
      LEFT JOIN vehicle_service_db.vehicle AS v ON a.vehicle_id = v.vehicle_id
      WHERE a.appointment_date >= NOW()
      ORDER BY a.appointment_date ASC;

Q: List clients with contact info.
SQL: SELECT c.customer_id, c.name, c.email, c.phone, c.created_at
      FROM customer_service_db.customer AS c
      ORDER BY c.created_at DESC;
"""

def generate_sql(question: str, customer_id: int = None) -> str:
    customer_filter = f"AND c.customer_id = {customer_id}" if customer_id else ""
    
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
 - If the question is vague like "about the service" or "service details",
   return an overview of services from service_management_db.service joined with
   employee_service_db.employee (assigned_to), vehicle_service_db.vehicle, and
   customer_service_db.customer, ordered by s.start_time DESC (NULLS LAST).
 - Use LEFT JOINs when related data may be missing (e.g., unassigned employee or
   vehicle without a linked customer).
  - Assume the user means the garage context even if they omit words like "garage" or "in the system".
  - Map common natural phrases to schemas:
    * "available services", "service types" → DISTINCT service_type from service_management_db.service
    * "service details", "about the service" → service overview with LEFT JOINs to employee/vehicle/customer
    * "staff", "team", "employees" → employee_service_db.employee
    * "cars", "vehicles" → vehicle_service_db.vehicle (join customer for owner name)
    * "bookings", "appointments" → appointment_service_db.appointment with joins
    * "clients", "customers" → customer_service_db.customer
  - When listing categories (like service types), prefer SELECT DISTINCT and ORDER BY ASC.
  - IMPORTANT: If a customer_id filter is provided, ALWAYS add it to the WHERE clause when querying customer-related data (appointments, vehicles, services related to customer vehicles).

{"CUSTOMER FILTER: Apply customer_id = " + str(customer_id) + " to all customer-specific queries (appointments, vehicles, services)." if customer_id else "NO CUSTOMER FILTER: Provide general information only (service types, employee list, general stats). DO NOT return specific appointment, vehicle, or service details."}

Examples:
{few_shot_examples}

Q: {question}
SQL:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    sql_query = response.choices[0].message.content.strip()
    return sql_query.replace("```sql", "").replace("```", "").strip()


def execute_sql(query: str):
    if not engine:
        return "Database connection is not configured."
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        print(f"SQL Execution Error: {e}")
        return f"Database Error: {e}"


def explain_results(question: str, df):
    if isinstance(df, str):
        return f"I couldn’t process your question: {df}"

    if df.empty:
        return "I couldn’t find any matching records for that question."

    total_rows = len(df)
    display_df = df.head(5)
    data_str = display_df.to_string(index=False)

    prompt = f"""
You are a helpful garage service assistant. Write a concise, friendly answer in perfect, natural, non-technical English. Do not mention SQL or database tables.

User question: "{question}"

Total result rows: {total_rows}
Preview of data (up to 5 rows):
{data_str}

Guidelines:
- If there is ONE row: write 2–3 sentences summarizing the key details (what service, status, when, who it’s assigned to, vehicle and customer if present, and any costs if present).
- If there are MULTIPLE rows: start with one sentence summarizing the overall picture (e.g., how many, upcoming items, common statuses). Then list 3–5 short bullet points highlighting representative items with: service type, status, date/time, assigned employee, vehicle (model + registration), and customer name when available.
- Use human terms (e.g., say "registration" for license_plate, "assigned to" for assigned_to). Format dates like "Nov 7, 2025 09:00". If a cost is present (estimated_cost/actual_cost), show it with LKR currency like "LKR 12,000".
- Be strictly factual from the data. Do not speculate. If a field is missing, just omit it.
- Do NOT include SQL, column names, or technical jargon. Keep under ~120 words.

Write the answer now.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def invoke_sql_chain(question: str, customer_id: int = None) -> str:
    """Invokes the full SQL chain (generate, execute, explain) and returns the answer."""
    try:
        sql_query = generate_sql(question, customer_id)
        print(f"Generated SQL: {sql_query}") 
        result_df = execute_sql(sql_query)
        reply = explain_results(question, result_df)
        return reply
    except Exception as e:
        print(f"Error in SQL chain: {e}")
        return f"Error processing SQL query: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_rag_system",
            "description": "Answers questions about Sri Lanka, Sri Lankan tourism, travel, places to visit, culture, and related topics. Use this for any general knowledge questions about Sri Lanka.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about Sri Lanka tourism."
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sql_system",
            "description": "Answers questions about the garage service, including customers, vehicles, appointments, service status, employees, and staff. Use this for any questions related to garage operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about the garage service."
                    }
                },
                "required": ["question"]
            }
        }
    }
]

def is_personal_data_request(question: str) -> bool:
    """
    Checks if the question is asking for personal/customer-specific data.
    Returns True if the question requests appointments, vehicles, services, or customer details.
    """
    personal_keywords = [
        'appointment', 'booking', 'my car', 'my vehicle', 'my service',
        'vehicle details', 'car details', 'service status', 'my booking',
        'when is my', 'status of my', 'my vehicles', 'my appointments'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in personal_keywords)

def classify_and_route(question: str, customer_id: int = None) -> str:
    """
    1. Classifies the user's question using OpenAI tool calling.
    2. Routes the question to the appropriate function (RAG or SQL).
    3. Returns the answer from that function.
    4. Validates customer_id for personal data requests.
    """
    
    # Check if customer is requesting personal data without providing customer_id
    if is_personal_data_request(question) and customer_id is None:
        return "Sorry! To view your appointment details, vehicle info, or service status, please log in to your account. But I can help you with general service information anytime!"
    
    system_prompt = """
    You are an intelligent routing assistant. Based on the user's question, you must call *one* of the provided functions.
    - Use `query_rag_system` for questions about Sri Lanka tourism, travel, or general knowledge.
    - Use `query_sql_system` for questions about the garage, customers, vehicles, appointments, or staff.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto" 
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
 
            if function_name == "query_rag_system":
                print(f"Routing to RAG system for question: {function_args['question']}")
                return invoke_rag_chain(function_args['question'])
                
            elif function_name == "query_sql_system":
                print(f"Routing to SQL system for question: {function_args['question']}")
                # Pass customer_id to SQL chain
                return invoke_sql_chain(function_args['question'], customer_id)
            
            else:
                return "Error: Unknown function call."

        print("No specific tool called. Defaulting to RAG system.")
        return invoke_rag_chain(question)

    except Exception as e:
        print(f"Error in classify_and_route: {e}")
        return "Sorry, I encountered an error trying to understand your request."