import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from google.cloud import bigquery
from openai import OpenAI

# Database schema 
DB_SCHEMAS = {
    "io_benchmark": """
    Table: io_benchmark
    Description: Table contenant des données de référence pour évaluer la performance des campagnes publicitaires.
    Columns:
    - group_object_field_id (INT): Identifiant unique du groupe de campagne publicitaire.
    - dsp (STRING): Plateforme publicitaire utilisée (ex. DV360).
    - kpi (STRING): Indicateur clé de performance (CTR, CPA, etc.).
    - advertiser_id (INT): Identifiant unique de l’annonceur.
    - client_id (INT): Identifiant du client associé à l’annonceur.
    - region (STRING): Région géographique de la campagne (ex., North America).
    - currency_code (STRING): Devise utilisée pour la campagne (USD, EUR, etc.).
    - revenue_usd_scibids (FLOAT): Revenu généré par la campagne en dollars USD.
    - media_cost_usd_scibids (FLOAT): Coût des médias en dollars USD.
    - total_media_cost_usd_scibids (FLOAT): Coût total des médias pour Scibids, y compris les frais supplémentaires.
    - revenue_usd_control (FLOAT): Revenu généré par la campagne du groupe témoin en USD.
    - media_cost_usd_control (FLOAT): Coût des médias pour le groupe témoin en USD.
    - total_media_cost_usd_control (FLOAT): Coût total des médias en USD pour le groupe témoin.
    - revenue_currency_scibids (FLOAT): Revenu généré par la campagne Scibids dans la devise locale.
    - media_cost_currency_scibids (FLOAT): Coût des médias pour la campagne Scibids dans la devise locale.
    - total_media_cost_currency_scibids (FLOAT): Coût total des médias pour la campagne Scibids dans la devise locale.
    - revenue_currency_control (FLOAT): Revenu généré par la campagne du groupe témoin dans la devise locale.
    - media_cost_currency_control (FLOAT): Coût des médias pour le groupe témoin dans la devise locale.
    - total_media_cost_currency_control (FLOAT): Coût total des médias en devise locale pour le groupe témoin.
    - media_cost_usd_scibids_complete_period (FLOAT): Coût des médias en USD pour Scibids sur toute la période complète.
    - media_cost_currency_scibids_complete_period (FLOAT): Coût des médias dans la devise locale pour Scibids sur toute la période complète.
    - revenue_usd_scibids_complete_period (FLOAT): Revenu en USD généré par la campagne Scibids sur toute la période complète.
    - revenue_currency_scibids_complete_period (FLOAT): Revenu dans la devise locale généré par la campagne Scibids sur toute la période complète.
    - kpi_value_scibids (FLOAT): Valeur du KPI pour la campagne Scibids.
    - kpi_value_control (FLOAT): Valeur du KPI pour le groupe témoin.
    - kpi_change (FLOAT): Changement dans la valeur du KPI entre le groupe témoin et la campagne Scibids.
    - uplift_percent (FLOAT): Pourcentage d'amélioration des performances de la campagne Scibids par rapport au groupe témoin.
    - fees_usd (FLOAT): Frais associés à la campagne en USD.
    - fees_currency (FLOAT): Frais associés à la campagne dans la devise locale.
    - fees_value (FLOAT): Valeur des frais appliqués à la campagne, en pourcentage.
    - roi (FLOAT): Retour sur investissement (ROI) de la campagne.
    - is_ab_test (BOOLEAN): Indique si la campagne fait partie d’un test A/B.
    - start_date_scibids (DATE): Date de début de la campagne Scibids.
    - end_date_scibids (DATE): Date de fin de la campagne Scibids.
    - start_date_control (DATE): Date de début du groupe témoin (si test A/B).
    - end_date_control (DATE): Date de fin du groupe témoin (si test A/B).
    - scibids_budget_portion (FLOAT): Proportion du budget allouée à Scibids.
    - media_cost_with_fees_usd_scibids (FLOAT): Coût total des médias avec frais pour Scibids en USD.
    - media_cost_with_fees_currency_scibids (FLOAT): Coût total des médias avec frais pour Scibids dans la devise locale.
    """,

    "new_benchmark": """
    Table: new_benchmark
    Description: Table contenant une vue d’ensemble de l’augmentation des performances pour chaque combinaison distincte de DSP / KPI.
    Columns:
    - dsp (STRING): Plateforme publicitaire utilisée (ex., DV360).
    - kpi (STRING): Indicateur clé de performance de la campagne (ex. CTR, CPA, CPM, VTR).
    - region (STRING): Région géographique associée à la campagne.
    - weighted_uplift (FLOAT): Amélioration pondérée des performances de la campagne.
    - elligible_ios (INT): Nombre de campagnes IO éligibles.
    - not_elligible_ios (INT): Nombre de campagnes IO non éligibles.
    - elligible_ab_test (INT): Nombre de campagnes incluses dans un test A/B.
    - not_elligible_ab_test (INT): Nombre de campagnes exclues d’un test A/B.
    """,

    "t_daily_io_features": """
    Table: t_daily_io_features
    Description: Table contenant des informations sur les fonctionnalités activées pour chaque IO par jour.
    Columns:
    - id_dsp_io_day_utc (INT): Identifiant unique de la journée DSP IO en UTC.
    - day_utc (DATE): Date en UTC.
    - dsp (STRING): Plateforme publicitaire utilisée.
    - sub_dsp (STRING): Sous-plateforme spécifique.
    - group_object_field_id (STRING): Identifiant du groupe d'objets.
    - kpi_to_optimize (STRING): KPI principal utilisé pour l'optimisation.
    - kpi_target (FLOAT): Objectif numérique du KPI optimisé.
    - min_cpm (FLOAT): Coût minimum autorisé.
    - max_cpm (FLOAT): Coût maximum autorisé.
    - remove_min_viz (BOOLEAN): Supprime le critère de visibilité.
    - force_pacing_asap_li (BOOLEAN): Active le pacing immédiat sur l'Insertion Order.
    - keep_trusted_inventory (BOOLEAN): Ne conserve que les inventaires de confiance.
    - use_custom_algo (BOOLEAN): Utilisation d'un algorithme personnalisé.
    """,

    "t_campaign_performance_day": """
    Table: t_campaign_performance_day
    Description: Table contenant des informations sur la performance des campagnes publicitaires.
    Columns:
    - dsp_li_flight_id (INT): Identifiant unique du flight DSP.
    - dsp_io_id (INT): Identifiant unique de l’Insertion Order.
    - dsp (STRING): Plateforme publicitaire utilisée.
    - impressions (INT): Nombre total d’impressions publicitaires.
    - media_cost_currency (FLOAT): Coût média dans la devise locale.
    - media_cost_usd (FLOAT): Coût média converti en USD.
    - revenue_currency (FLOAT): Revenu généré dans la devise locale.
    - revenue_usd (FLOAT): Revenu généré en USD.
    - clicks (INT): Nombre de clics enregistrés.
    - impressions_viewed (FLOAT): Nombre d’impressions effectivement vues.
    """,

    "t_flights": """
    Table: t_flights
    Description: Table contenant des informations sur le suivi d’une OI/OF.
    Columns:
    - id_dsp_li_flight (INT): Identifiant unique du vol DSP.
    - dsp (STRING): Plateforme publicitaire utilisée.
    - deleted (BOOLEAN): Indique si le vol a été supprimé.
    - gof_flight_id (INT): Identifiant du vol GOF.
    - of_flight_id (INT): Identifiant du vol OF.
    - timezone (STRING): Fuseau horaire du vol.
    - gof_flight_budget_money (FLOAT): Budget monétaire du vol GOF.
    - of_flight_budget_imp (INT): Budget en impressions du vol OF.
    """
}


OPENAI_API_KEY = "****" # Add your OpenAI API key here

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Set Google Cloud project ID
PROJECT_ID = "capstone-448012"

# Initialize BigQuery Client
client_bq = bigquery.Client(project=PROJECT_ID)
# 🔹 Step 1: Understand the User’s Request

def classify_query_with_agent(user_input):
    """
    Uses an AI agent to classify the user request into:
    - "answer" (general response)
    - "execute_sql" (generate SQL)
    - "visualize" (generate SQL + plot)
    - Detects the type of visualization (bar, pie, histogram, scatter, line chart, etc.)
    """

    # Convert schema dictionary into a readable format for AI
    schema_description = "\n".join([f"Table: {key}\n{value}" for key, value in DB_SCHEMAS.items()])

    system_prompt = f"""
    You are an AI assistant that classifies user queries into one of the following categories:
    - "answer": If the query is a general question that does not involve database queries or visualization.
    - "execute_sql": If the query asks for data retrieval, computation, or SQL execution.
    - "visualize": If the query requires a graph, chart, histogram, or other data visualization.

    **Available Database Schema:**
    {schema_description}

    **Visualization Types to Detect**:
    - "bar_chart": For grouped categorical comparisons.
    - "pie_chart": When the user asks for proportional data distribution.
    - "histogram": When the user asks for a histogram (often with a density curve).
    - "line_chart": If the user wants a trend over time.
    - "scatter_plot": If comparing two numerical values.
    
    **Examples of Classification**:
    - "What is a DSP?" → {{"task": "answer"}}
    - "Get the average uplift for each DSP" → {{"task": "execute_sql"}}
    - "Plot the histogram of average weighted uplift per DSP" → {{"task": "visualize", "chart_type": "histogram"}}
    - "Draw the number of campaigns by region" → {{"task": "visualize", "chart_type": "bar_chart"}}
    - "Show a pie chart of revenue distribution per advertiser" → {{"task": "visualize", "chart_type": "pie_chart"}}
    - "Give the SQL query that returns the number of campaigns by region" → {{"task": "execute_sql"}}
    - "Fetch the data and then plot the trends as a line chart" → {{"task": "visualize", "chart_type": "line_chart"}}
    - "Retrieve and analyze revenue growth over time with a scatter plot" → {{"task": "visualize", "chart_type": "scatter_plot"}}

    **Output strictly in JSON format:**
    {{"task": "<answer | execute_sql | visualize>", "chart_type": "<bar_chart | pie_chart | histogram | line_chart | scatter_plot | none>"}}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Call OpenAI GPT-4
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    try:
        # Extract the JSON response
        classification = json.loads(response.choices[0].message.content.strip())

        # Ensure valid classification
        if classification["task"] in ["answer", "execute_sql", "visualize"]:
            return classification
        else:
            return {"task": "execute_sql", "chart_type": "none"}  # Default to SQL execution if uncertain

    except Exception as e:
        print(f"Classification Error: {e}")
        return {"task": "execute_sql", "chart_type": "none"}  # Default to SQL execution if error occurs


def clean_sql_query(sql_output):
    """Extracts SQL query and replaces table names with `data.<table_name>`."""
    
    # Remove markdown formatting (```sql and ```)
    sql_cleaned = re.sub(r"```sql|```", "", sql_output).strip()

    # Replace table names with "data.<table_name>"
    for table in DB_SCHEMAS.keys():
        sql_cleaned = sql_cleaned.replace(f"{table}", f"data.{table}")

    return sql_cleaned

def execute_bigquery(sql_query):
    """Executes an SQL query on BigQuery and returns results if successful."""
    try:
        if not sql_query.strip():
            st.error("❌ Generated SQL query is empty! Please check the query generation process.")
            return None

        st.subheader("🔍 SQL Query Sent to BigQuery:")
        st.code(sql_query, language="sql")  # Display the query before execution

        # Check query validity (Dry Run)
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = client_bq.query(sql_query, job_config=job_config)
        st.success("✅ Query is valid! Running execution...")

        # Execute query if valid
        query_job = client_bq.query(sql_query)
        results = query_job.result().to_dataframe()

        if results.empty:
            st.warning("⚠️ Query executed successfully but returned no results.")
            return None
        else:
            st.success("✅ Query executed successfully!")
            return results

    except Exception as e:
        st.error(f"❌ Query failed: {e}")
        return None


def recognize_tables(user_query):
    """Identify relevant tables from user input."""
    tables_description = "\n".join([f"Table: {key}\n{value}" for key, value in DB_SCHEMAS.items()])
    
    messages = [
        {"role": "system", "content": "You are an expert in SQL database structures. Given a user request, identify the most relevant tables from the schema provided that will be called in the SQL query."},
        {"role": "user", "content": f"Available Database Tables:\n{tables_description}\n\nUser Request:\n{user_query}\n\nReturn only the relevant table names, separated by commas."}
    ]
    
    response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.2)
    return response.choices[0].message.content.strip().split(", ")

# Function to generate an SQL query based on the instruction
def generate_sql(user_query,selected_tables):

    """Generate an SQL query using the selected tables."""

    # Define keys for each table (used for joins & conditions)
    keys_identification = {
        "io_benchmark": ["group_object_field_id", "dsp", "kpi"],
        "new_benchmark": ["kpi", "dsp"],
        "t_daily_io_features": ["dsp", "group_object_field_id", "day_utc"],
        "t_flights": ["group_object_field_id", "dsp", "object_field_id", "gof_flight_id", "of_flight_id"],
        "t_insertion_orders": ["group_object_field_id", "dsp"],
        "t_campaign_performance_day": ["group_object_field_id", "dsp", "object_field_id", "gof_flight_id", "date_hour_tz"],
        "t_reach_performance": ["group_object_field_id", "dsp", "end_date"],
        "t_pixel_performance": ["group_object_field_id", "object_field_id", "dsp", "conversion_pixel_id", "date_hour_tz"],
    }

    # Extract only the relevant keys for the selected tables
    relevant_keys = "\n".join([f"{table}: {', '.join(keys_identification[table])}" for table in selected_tables if table in keys_identification])
    relevant_schemas = "\n\n".join([DB_SCHEMAS[table.strip()] for table in selected_tables if table.strip() in DB_SCHEMAS])
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert SQL assistant specialized in digital advertising and data analytics. "
                "Generate only the SQL query while strictly following the provided database schemas and user request.\n"
                "Keys Identification (for proper joins and filtering):\n"
                f"{relevant_keys}\n\n"
                f"Database Schema:\n{relevant_schemas}"
            ),
        },
        {
            "role": "user",
            "content": f"Database Schema:\n{relevant_schemas}\n\nInstruction:\n{user_query}\n\nGenerate only the SQL query without explanation.",
        },
    ]


    # Call OpenAI API to generate the SQL query
    completion = client.chat.completions.create(
        model="gpt-4o",  # High-performance model for SQL generation
        messages=messages,
        temperature=0.2,  # Low temperature for deterministic output
    )

    return completion.choices[0].message.content

# 🔹 Step 5: Plot Results

def generate_visualization_code(user_prompt, results):
    """
    Dynamically generates **Python visualization code** based on the actual SQL query results.
    Ensures correct column selection and uses **all available data**.
    """

    # Ensure results exist and are valid
    if results is None or results.empty:
        return "st.error('⚠️ No data available for visualization.')"
    dataset_json = results.to_json(orient='records')

    # Get column types
    numerical_cols = results.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = results.select_dtypes(exclude=["number"]).columns.tolist()

    # Ensure valid columns exist
    if not numerical_cols or not categorical_cols:
        return "st.error('⚠️ No valid columns found for visualization.')"

    # Default column selections (adjusted dynamically)
    x_col = categorical_cols[0]  # Typically 'client_id' or 'region'
    y_col = numerical_cols[0]  # Typically 'revenue' or 'cost'

    # **Limit data to 50 rows to avoid overcrowding the visualization**
    results = results.head(50)

    system_prompt = f"""
    You are a Python visualization assistant that generates **Matplotlib/Seaborn plots for Streamlit**.
    
    **User Request:** "{user_prompt}"
    **Full Dataset (JSON Format):** {dataset_json}
    
    **Rules for Code Generation:**
    - **Use results exactly as provided**, do not modify or truncate the data.
    - **Limit to 50 rows** to avoid excessive plotting.
    - **Ensure large numbers are formatted** (e.g., currency should be formatted properly).
    - **Use correct columns**:
      - Categorical for `x`: `{x_col}`
      - Numerical for `y`: `{y_col}`
    - **Rotate x-axis labels if needed** to prevent overlap.
    - **Ensure `st.pyplot(fig)` is included** for rendering.
    - **Always return valid Python code with no syntax errors.**
    - **Escape special characters properly to prevent execution errors.**

    **Expected Output Example:**
    ```python
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=results, x="{x_col}", y="{y_col}", ax=ax)
    ax.set_title("{y_col} by {x_col}")
    ax.set_xlabel("{x_col}")
    ax.set_ylabel("{y_col}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)
    ```
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    response_text = response.choices[0].message.content.strip()

    # Debugging: Show raw OpenAI response
    st.code(response_text, language="python")

    if not response_text:
        return "st.error('⚠️ Failed to generate visualization code.')"

    # Remove markdown formatting if included
    if response_text.startswith("```python"):
        response_text = response_text[9:-3].strip()

    return response_text


def visualize_data(visualization_code, results):
    """
    Dynamically executes the generated Python visualization code using Matplotlib/Seaborn.
    """
    if results is None or results.empty:
        st.warning("⚠️ No data available for visualization.")
        return

    # Define a controlled execution environment
    execution_globals = {
        "results": results,
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "st": st
    }

    # Execute the generated code within the controlled scope
    try:
        exec(visualization_code, execution_globals)
        st.success("✅ Visualization executed successfully!")
    except Exception as e:
        st.error(f"❌ An error occurred while executing visualization: {str(e)}")

def intelligent_agent(user_prompt):
    """
    Handles complex multi-step queries (SQL generation, visualization, and analysis).
    It detects multiple required tasks from a single user request and structures execution order.
    """

    system_prompt = f"""
    You are an AI that breaks down multi-step data requests into **structured tasks**.

    **User Request:** "{user_prompt}"

    **Task Breakdown Rules:**
    - **Extract individual tasks** from the request (SQL query, visualizations, analysis).
    - **Ensure SQL queries run first**, as visualizations/analysis depend on data.
    - If multiple visualizations are requested, **generate distinct descriptions** for each.
    - If analysis is required, **interpret data** and answer the user's question concisely.
    - **DO NOT return actual SQL code. Instead, describe what the query should do**.

    **Example Breakdown:**
    **User Query:** *"Get the revenue per region, plot a bar chart, and summarize the insights."*
    **Decomposed Tasks:**
    [
        {{"task": "sql", "content": "Generate an SQL query to compute the total revenue per region from the sales data."}},
        {{"task": "visualization", "content": "Generate a bar chart of revenue per region."}},
        {{"task": "analysis", "content": "Summarize the revenue distribution across regions."}}
    ]

    **Expected JSON Output Format (Strictly Without Markdown Formatting):**
    [
        {{"task": "<task_type>", "content": "<task description>"}}
    ]
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    # Extract response text
    response_text = response.choices[0].message.content.strip()

    # **🚨 Remove Markdown Formatting (if present)**
    response_text = re.sub(r"^```json|```$", "", response_text).strip()

    # **🔍 Debugging: Show AI Response Before Parsing**
    st.subheader("🔍 AI Task Breakdown Debug:")
    st.text(response_text)  # Show raw response

    # **🚨 Parse JSON Safely**
    try:
        parsed_response = json.loads(response_text)
        return parsed_response if isinstance(parsed_response, list) else [parsed_response]
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Parsing Error: {str(e)}")
        return [{"task": "error", "content": "Invalid JSON response from OpenAI."}]

def analysis_agent(results, analysis_task_content):
    """
    Uses AI to generate a **direct answer** to the user's request based on the provided results.

    Parameters:
    - results (pd.DataFrame): The DataFrame containing the SQL query results.
    - analysis_task_content (str): The specific question or request provided for analysis.

    Returns:
    - A dictionary with the AI-generated response.
    """

    # Ensure results exist
    if results is None or results.empty:
        return {"task": "error", "content": "No data available to answer the question."}

    # Convert the results DataFrame to JSON format for AI processing
    data_for_ai = results.to_dict(orient="records")

    system_prompt = f"""
    You are a data assistant specialized in answering user questions **based on structured query results**.

    **Task:** Answer the user's request using the provided dataset.

    **User Request:** "{analysis_task_content}"

    **Available Data (First 50 rows shown):**
    ```json
    {json.dumps(data_for_ai[:50], indent=2)}
    ```

    **Guidelines for Response:**
    - **Answer only what was asked.** Do not generate SQL queries or visualization code.
    - **Base your response only on the provided dataset.** Do not make assumptions.
    - **If the question requires calculations, compute them from the dataset.**
    - **If a specific insight is requested, explain it using the actual data.**
    - **If multiple tasks are requested, break them down into clear answers.**
    - **Be concise, relevant, and accurate.**

    **Response Format (Strict JSON Only):**
    ```json
    {{"task": "analysis", "content": "<AI-generated answer>"}}
    ```
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": analysis_task_content}
    ]

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )
    response_text = response.choices[0].message.content.strip()

    # Ensure response is valid JSON
    if response_text.startswith("```json"):
        response_text = response_text[7:-3].strip()

    try:
        parsed_response = json.loads(response_text)  # Convert string to dictionary

        # Check if the response is valid and formatted correctly
        if isinstance(parsed_response, dict) and parsed_response.get("task") == "analysis":
            st.subheader("📢 Analysis & Insights")
            st.write(parsed_response["content"])  # Display AI-generated response
            return parsed_response

        return {"task": "error", "content": "Invalid AI response format."}

    except json.JSONDecodeError:
        return {"task": "error", "content": "Failed to parse AI response as JSON."}
