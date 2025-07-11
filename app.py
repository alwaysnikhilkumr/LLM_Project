import streamlit as st
import tempfile
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Page configuration
st.set_page_config(
    page_title="SQL LLM Query Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .sql-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'sql_chain' not in st.session_state:
    st.session_state.sql_chain = None
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False

# Title
st.markdown('<h1 class="main-header">🔍 SQL LLM Query Assistant</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Google API Key
    st.subheader("🔑 Google API Key")
    api_key = st.text_input(
        "Enter your Google API Key:",
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    # Database Configuration
    st.subheader("🗄️ Database Configuration")
    db_host = st.text_input("Host", value="localhost")
    db_port = st.number_input("Port", value=5432, min_value=1, max_value=65535)
    db_name = st.text_input("Database Name", value="SQL_Project")
    db_user = st.text_input("Username", value="postgres")
    db_password = st.text_input("Password", type="password", value="PASSWORD")
    
    # Connect button
    if st.button("🔌 Connect to Database", type="primary"):
        if api_key and all([db_host, db_port, db_name, db_user, db_password]):
            try:
                with st.spinner("Connecting to database and initializing AI..."):
                    # Initialize LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=api_key,
                        temperature=0.1
                    )
                    
                    # Database connection
                    db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                    db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
                    
                    # Few-shot examples
                    few_shots = [
                        {
                            'Question': "How many total orders do we have?",
                            'SQLQuery': "SELECT COUNT(*) as total_orders FROM orders;",
                            'SQLResult': "[(1500,)]",
                            'Answer': "We have 1500 total orders in the database."
                        },
                        {
                            'Question': "What are the top 5 selling products by quantity?",
                            'SQLQuery': "SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id, p.name ORDER BY total_sold DESC LIMIT 5;",
                            'SQLResult': "[('Product_A', 500), ('Product_B', 450)]",
                            'Answer': "The top 5 selling products by quantity are listed above."
                        },
                        {
                            'Question': "Show me the total revenue by region",
                            'SQLQuery': "SELECT region, SUM(total_amount) as total_revenue FROM orders GROUP BY region ORDER BY total_revenue DESC;",
                            'SQLResult': "[('North', 150000), ('South', 120000)]",
                            'Answer': "The total revenue by region is shown above, with North region leading."
                        },
                        {
                            'Question': "Which customers have placed the most orders?",
                            'SQLQuery': "SELECT customer_name, COUNT(*) as order_count FROM orders GROUP BY customer_id, customer_name ORDER BY order_count DESC LIMIT 10;",
                            'SQLResult': "[('John Doe', 25), ('Jane Smith', 20)]",
                            'Answer': "The customers with the most orders are listed above."
                        },
                        {
                            'Question': "What is the average order value?",
                            'SQLQuery': "SELECT AVG(total_amount) as average_order_value FROM orders;",
                            'SQLResult': "[(85.50,)]",
                            'Answer': "The average order value is $85.50."
                        }
                    ]
                    
                    # Initialize embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    
                    # Create texts for vectorization
                    texts = []
                    for example in few_shots:
                        text = f"Question: {example['Question']} Answer: {example['Answer']}"
                        texts.append(text)
                    
                    # Create temporary directory for ChromaDB
                    temp_dir = tempfile.mkdtemp()
                    
                    # Create vector store
                    vectorstore = Chroma.from_texts(
                        texts=texts,
                        embedding=embeddings,
                        metadatas=few_shots,
                        persist_directory=temp_dir
                    )
                    
                    # Create example selector
                    example_selector = SemanticSimilarityExampleSelector(
                        vectorstore=vectorstore,
                        k=2
                    )
                    
                    # Define example prompt template
                    example_prompt = PromptTemplate(
                        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
                        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
                    )
                    
                    # Create few-shot prompt template
                    few_shot_prompt = FewShotPromptTemplate(
                        example_selector=example_selector,
                        example_prompt=example_prompt,
                        prefix="""You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run.
Unless the user specifies a specific number of examples, query for at most 10 results using LIMIT clause.
Never query for all columns from a table. You must query only the columns needed to answer the question.
Pay attention to use only the column names you can see in the tables below.
Be very careful about which column to use for filtering and make sure the column name exists in the table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Here are some examples:""",
                        suffix="\nQuestion: {input}\nSQLQuery:",
                        input_variables=["input", "table_info"]
                    )
                    
                    # Create SQL chain
                    sql_chain = SQLDatabaseChain.from_llm(
                        llm=llm,
                        db=db,
                        prompt=few_shot_prompt,
                        verbose=True,
                        return_intermediate_steps=True
                    )
                    
                    # Store in session state
                    st.session_state.sql_chain = sql_chain
                    st.session_state.db = db
                    st.session_state.db_connected = True
                    
                    st.success("✅ Successfully connected to database and initialized AI!")
                    
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                st.session_state.db_connected = False
        else:
            st.error("❌ Please fill in all required fields")
    
    # Connection status
    if st.session_state.db_connected:
        st.markdown('<div class="success-box">🟢 Database Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">🔴 Database Not Connected</div>', unsafe_allow_html=True)
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Your Question")
    
    # Question input
    user_question = st.text_area(
        "Enter your question about the database:",
        height=100,
        placeholder="e.g., How many orders were placed last month?",
        help="Ask questions in natural language about your data"
    )
    
    # Example questions
    st.subheader("💡 Example Questions")
    example_questions = [
        "How many total orders do we have?",
        "What are the top 5 selling products?",
        "Show me the total revenue by region",
        "Which customers have placed the most orders?",
        "What is the average order value?",
        "Which products need to be reordered?"
    ]
    
    # Create buttons for example questions
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📝 {question}", key=f"example_{i}"):
                user_question = question
                st.rerun()
    
    # Submit button
    if st.button("🚀 Get Answer", type="primary", disabled=not st.session_state.db_connected):
        if user_question and st.session_state.sql_chain:
            try:
                with st.spinner("🤔 Thinking and querying database..."):
                    result = st.session_state.sql_chain(user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': result['result'],
                        'sql': result['intermediate_steps'][0]['sql_cmd'] if result.get('intermediate_steps') else None
                    })
                    
                    st.success("✅ Query executed successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': f"Error: {str(e)}",
                    'sql': None
                })

with col2:
    st.header("📊 Database Information")
    
    if st.session_state.db_connected and hasattr(st.session_state, 'db'):
        with st.expander("🗂️ Database Schema", expanded=False):
            st.code(st.session_state.db.table_info, language="sql")
        
        with st.expander("📈 Quick Stats", expanded=True):
            try:
                # Get table names
                table_names = st.session_state.db.get_usable_table_names()
                st.write(f"**Total Tables:** {len(table_names)}")
                
                for table in table_names[:5]:  # Show first 5 tables
                    st.write(f"• {table}")
                
                if len(table_names) > 5:
                    st.write(f"... and {len(table_names) - 5} more tables")
                    
            except Exception as e:
                st.write("Unable to fetch database stats")
    else:
        st.info("Connect to database to see schema information")

# Chat History
if st.session_state.chat_history:
    st.header("💭 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
        with st.container():
            st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
            
            if chat.get('sql'):
                with st.expander("🔍 View SQL Query"):
                    st.markdown(f'<div class="sql-box">{chat["sql"]}</div>', unsafe_allow_html=True)
            
            if chat['answer'].startswith('Error:'):
                st.markdown(f'<div class="error-box">**Answer:** {chat["answer"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"**Answer:** {chat['answer']}")
            
            st.markdown("---")

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🔍 SQL LLM Query Assistant | Built with Streamlit & LangChain</p>
    <p>💡 Tip: Be specific with your questions for better results</p>
</div>
""", unsafe_allow_html=True)