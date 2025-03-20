import faiss
import numpy as np
import pickle
import google.generativeai as genai
import os
import asyncio
import streamlit as st
from openai import OpenAI


OPENAI_API_KEY = "*****"
# Base de données des schémas (complète)
DB_SCHEMAS = {
    "io_benchmark": {
        "group_object_field_id": "Unique identifier for the campaign group.",
        "dsp": "Advertising platform used (e.g., DV360).",
        "kpi": "Key performance indicator (CTR, CPA, etc.).",
        "advertiser_id": "Unique identifier for the advertiser.",
        "client_id": "Unique identifier for the client associated with the advertiser.",
        "region": "Geographical region of the campaign (e.g., North America).",
        "currency_code": "Currency used for the campaign (USD, EUR, etc.).",
        "total_media_cost_usd_scibids": "Total media cost in USD for Scibids, including additional fees.",
        "total_media_cost_usd_control": "Total media cost in USD for the control group, including fees.",
        "kpi_value_scibids": "Key performance indicator (KPI) value for the Scibids campaign.",
        "kpi_value_control": "Key performance indicator (KPI) value for the control group.",
        "kpi_change": "Change in KPI value between the control group and the Scibids campaign.",
        "uplift_percent": "Percentage improvement in campaign performance for Scibids compared to the control group.",
        "fees_usd": "Fees associated with the campaign in USD.",
        "fees_value": "Value of the fees applied to the campaign, in percentage.",
        "roi": "Return on investment (ROI) of the campaign.",
        "is_ab_test": "Indicates whether the campaign is part of an A/B test (True/False).",
        "start_date_scibids": "Start date of the Scibids campaign.",
        "end_date_scibids": "End date of the Scibids campaign.",
        "start_date_control": "Start date of the control group (if A/B test).",
        "end_date_control": "End date of the control group (if A/B test).",
        "scibids_budget_portion": "Portion of the budget allocated to Scibids."
    },
    "new_benchmark": {
        "dsp": "Advertising platform used (e.g., DV360).",
        "kpi": "Key performance indicator of the campaign (e.g., CTR, CPA, CPM, VTR).",
        "region": "Geographic region associated with the campaign (may contain null values).",
        "weighted_uplift": "Weighted performance improvement of the campaign, measured by the relative change in the selected KPI.",
        "elligible_ios": "Number of insertion orders (IOs) eligible for uplift analysis.",
        "not_elligible_ios": "Number of IOs not eligible for uplift analysis.",
        "elligible_ab_test": "Number of campaigns included in an A/B test.",
        "not_elligible_ab_test": "Number of campaigns excluded from an A/B test."
    },
    "t_daily_io_features": {
        "id_dsp_io_day_utc": "Unique identifier for the DSP IO day in UTC.",
        "day_utc": "Corresponding date in UTC.",
        "dsp": "Advertising platform used (e.g., DV360, MediaMath).",
        "sub_dsp": "Sub-platform or specific category within the DSP.",
        "group_object_field_id": "Identifier for the campaign group.",
        "kpi_to_optimize": "Main KPI used for optimization.",
        "kpi_target": "Numerical target for the optimized KPI.",
        "min_cpm": "Minimum cost per thousand impressions allowed.",
        "max_cpm": "Maximum cost per thousand impressions allowed.",
        "min_margin": "Minimum margin allowed for the campaign.",
        "min_viz": "Minimum required visibility level.",
        "remove_min_viz": "Indicates whether the minimum visibility criterion should be ignored.",
        "overwrite_frequency": "If true, forces the rewriting of frequency rules.",
        "log_day": "Data logging date.",
        "overwrite_creative_selection": "If true, forces manual selection of ad creatives.",
        "overwrite_li_budget": "Indicates whether the budget for the insertion order should be forced.",
        "force_pacing_asap_li": "Enables immediate pacing on the insertion order.",
        "keep_trusted_inventory": "If true, retains only trusted inventories.",
        "remove_budget_strat_imp": "Removes certain strategic budget constraints.",
        "force_pacing_asap_imp": "Immediately applies pacing to the impression.",
        "use_custom_algo": "Activates the use of a custom optimization algorithm."
    },
    "t_campaign_performance_day": {
        "client_id": "Unique identifier for the client associated with the campaign.",
        "dsp": "Name of the advertising platform (e.g., DV360, Xandr, MediaMath).",
        "group_object_field_id": "Identifier for grouped advertising objects.",
        "object_field_id": "Unique identifier for the advertising object.",
        "gof_flight_id": "Flight ID for the group object.",
        "of_flight_id": "Flight ID for a specific object.",
        "billing_scibids_activity": "Defines whether Scibids was active on the campaign at a given time.",
        "day_tz": "Date with local timezone.",
        "day_utc": "Date in UTC.",
        "impressions": "Total number of ad impressions.",
        "media_cost_currency": "Media cost in local currency.",
        "media_cost_usd": "Media cost converted to USD.",
        "revenue_currency": "Revenue generated in local currency.",
        "revenue_usd": "Revenue generated in USD.",
        "clicks": "Number of recorded clicks.",
        "impressions_viewed": "Number of impressions actually viewed.",
        "impressions_viewed_measured": "Number of measured impressions.",
        "completed_view": "Number of completed views.",
        "audible_viewable_completed": "Number of completed views with audio enabled.",
        "profit_currency": "Profit generated in local currency.",
        "profit_usd": "Profit generated in USD.",
        "total_cost_currency": "Total cost in local currency.",
        "total_cost_usd": "Total cost in USD.",
        "trueview_views": "Number of TrueView views.",
        "youtube_conversions": "Number of YouTube conversions."
    },
    "t_flights": {
        "id_dsp_li_flight": "Unique identifier for the DSP flight.",
        "dsp": "Advertising platform used.",
        "sub_dsp": "Sub-platform used.",
        "group_object_field_id": "Group object identifier.",
        "object_field_id": "Object identifier.",
        "deleted": "Indicates whether the flight has been deleted.",
        "gof_flight_id": "GOF flight identifier.",
        "of_flight_id": "OF flight identifier.",
        "flight_billing_activated": "Indicates whether Scibids was active on this flight.",
        "flight_activity": "Flight status (completed, ongoing, or future).",
        "gof_start_date_of_tz": "GOF start date (timezone).",
        "gof_end_date_of_tz": "GOF end date (timezone).",
        "gof_flight_budget_money": "Monetary budget for the GOF flight.",
        "of_start_date_of_tz": "OF start date (timezone).",
        "of_end_date_of_tz": "OF end date (timezone)."
    },
    "t_fit_scores_daily": {
        "id_dsp_io_day": "Unique identifier of the IO per day.",
        "dsp": "Advertising platform used.",
        "sub_dsp": "Sub-platform used.",
        "day": "Date of the fit score.",
        "group_object_field_id": "Identifier of the group of objects.",
        "final_score": "Final calculated score."
    },
    "t_insertion_orders": {
        "dsp": "Name of the demand-side platform (DSP).",
        "sub_dsp": "Identifier of the sub-DSP.",
        "parent_object_field_id": "Identifier of the parent object.",
        "group_object_field_id": "Identifier of the group of objects.",
        "member_id": "Unique identifier of the associated member.",
        "advertiser_id": "Unique identifier of the advertiser.",
        "advertiser_name": "Name of the advertiser.",
        "timezone": "Associated timezone.",
        "currency_code": "Code of the currency used.",
        "exchange_rate": "Applied exchange rate.",
        "surcouche_setup": "Indicates if an overlay is activated.",
        "keystone_status": "Describes Scibids activity status on the campaign.",
        "addressability": "Determines if the IO is optimizable (addressable) or not.",
        "ab_test_start_date": "Start date of the A/B test.",
        "ab_test_end_date": "End date of the A/B test."
    },
    "t_reach_performance": {
        "id_dsp_io_day": "Unique identifier of the row per day and DSP.",
        "dsp": "Name of the demand-side platform (DSP).",
        "insertion_date": "Date and time of data insertion.",
        "client_id": "Unique identifier of the client.",
        "start_date": "Start date of the campaign.",
        "end_date": "End date of the campaign.",
        "group_object_field_id": "Unique identifier of the group of objects.",
        "unique_reach_total": "Total number of unique users reached.",
        "impressions": "Total number of recorded impressions.",
        "prev_value": "Previous value of unique reach.",
        "last_new_unique_reach": "Latest newly reached unique value.",
        "last_new_reach_value": "Indicator if the latest reach value is valid."
    },
    "t_pixel_performance": {
        "dsp": "Name of the demand-side platform (DSP).",
        "sub_dsp": "Identifier of the sub-DSP.",
        "client_id": "Unique identifier of the client.",
        "group_object_field_id": "Unique identifier of the group of objects.",
        "conversion_pixel_id": "Identifier of the conversion pixel.",
        "conversion_pixel_name": "Name of the conversion pixel.",
        "object_field_id": "Identifier of the associated object field.",
        "day_tz": "Date of the day with local timezone.",
        "day_utc": "Date of the day in UTC.",
        "date_hour_tz": "Hour of the event with local timezone.",
        "date_hour_utc": "Hour of the event in UTC.",
        "pixel_type": "Type of pixel used.",
        "billing_scibids_activity": "Indicates whether Scibids was active at the time of conversion.",
        "pixel_origin": "Origin of the pixel.",
        "pixel_post_click_convs": "Number of post-click conversions.",
        "pixel_post_view_convs": "Number of post-view conversions.",
        "pixel_total_convs": "Total number of conversions."
    }
}
# Set up Google API Key
os.environ["GOOGLE_API_KEY"] = "******"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure the event loop is set up correctly
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Step 1: Function to Generate Embeddings Using Google's `embedding-001`
def get_gemini_embedding(text):
    """Generates an embedding vector using Google's `embedding-001` model."""
    response = genai.embed_content(
        model="models/embedding-001",  # ✅ Correct model name
        content=text,
        task_type="retrieval_document"  # Task type is required
    )
    return np.array(response["embedding"])  # Convert to NumPy array


# ✅ Step 3: Generate Text Descriptions for Embeddings
texts = []
metadata = []
for table_name, columns in DB_SCHEMAS.items():
    for col_name, col_desc in columns.items():
        text = f"{col_name} in table {table_name}: {col_desc}"
        texts.append(text)
        metadata.append((table_name, col_name))

# ✅ Step 4: Generate Embeddings Using Gemini
embeddings = np.array([get_gemini_embedding(text) for text in texts])

# ✅ Step 5: Create and Store FAISS Index
embedding_dim = embeddings.shape[1]  # Get embedding size
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
index.add(embeddings)  # Add embeddings to FAISS

# ✅ Step 6: Save FAISS Index, Metadata & Embeddings
faiss.write_index(index, "schema_index.faiss")
with open("schema_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
np.save("schema_embeddings.npy", embeddings)

print("✅ FAISS Index Created & Saved Successfully!")

# ✅ Step 7: Function to Load FAISS Index & Metadata
def load_faiss_index():
    index = faiss.read_index("schema_index.faiss")
    with open("schema_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    embeddings = np.load("schema_embeddings.npy")
    return index, metadata, embeddings

# ✅ Step 8: Function to Search Relevant Columns
def search_relevant_columns(query, top_k=50):
    index, metadata, _ = load_faiss_index()  # Load FAISS index & metadata
    query_embedding = get_gemini_embedding(query).reshape(1, -1)  # Get query embedding
    D, I = index.search(query_embedding, top_k)  # Search FAISS index
    results = [metadata[i] for i in I[0] if i != -1]  # Retrieve matching columns
    return results

# ✅ Step 9: Function to Generate SQL Query Using OpenAI
def generate_sql_with_llm(query):
    relevant_columns = search_relevant_columns(query)

    # Extract relevant tables and columns
    tables = sorted(set(table for table, _ in relevant_columns))  # Sort for better readability
    columns = ", ".join([col for _, col in relevant_columns]) if relevant_columns else "*"

    prompt = f"""
    Generate only an optimized SQL query based on the user's request.
    Do not include any explanations, assumptions, or comments.

    - User request: "{query}"
    - Relevant tables: {", ".join(tables)}
    - Relevant columns: {columns}

    Ensure the query is optimized and syntactically correct. If multiple tables are involved, use JOINs appropriately.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an SQL assistant who writes optimized SQL queries. Only return the SQL query itself, without any explanations or assumptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Less randomness for consistent SQL structure
        max_tokens=250
    )

    return response.choices[0].message.content.strip()

# ✅ Step 10: Streamlit UI for SQL Query Generation
st.title("AI-Powered SQL Query Generator")

# User input for query
user_query = st.text_input("Enter your query:", placeholder="What is the total amount spent by the client ID_SELECTED?")

# Process the query when the user submits
if st.button("Generate SQL Query"):
    if user_query:
        with st.spinner("Generating SQL query..."):
            sql_query = generate_sql_with_llm(user_query)
            st.code(sql_query, language="sql")  # ✅ Displays ONLY the SQL query
    else:
        st.warning("Please enter a query.")
