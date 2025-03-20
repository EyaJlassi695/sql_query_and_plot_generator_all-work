import re

import streamlit as st
from google.cloud import bigquery
from openai import OpenAI

# Database schema 
DB_SCHEMAS = {
    "io_benchmark": """
    Table: io_benchmark
    Description: Reference data table for evaluating advertising campaign performance. 
    Each row represents a "before/after" test or an A/B test conducted in the past. 
    Not all campaigns are necessarily present in this table.

    Columns:
    - group_object_field_id (INT): Unique identifier for the campaign group.
    - dsp (STRING): Advertising platform used (e.g., DV360).
    - kpi (STRING): Key performance indicator (CTR, CPA, etc.).
    - advertiser_id (INT): Unique identifier for the advertiser.
    - client_id (INT): Unique identifier for the client associated with the advertiser.
    - region (STRING): Geographical region of the campaign (e.g., North America).
    - currency_code (STRING): Currency used for the campaign (USD, EUR, etc.).
    - total_media_cost_usd_scibids (FLOAT): Total media cost in USD for Scibids, including additional fees.
    - total_media_cost_usd_control (FLOAT): Total media cost in USD for the control group, including fees.
    - kpi_value_scibids (FLOAT): Key performance indicator (KPI) value for the Scibids campaign.
    - kpi_value_control (FLOAT): Key performance indicator (KPI) value for the control group.
    - kpi_change (FLOAT): Change in KPI value between the control group and the Scibids campaign.
    - uplift_percent (FLOAT): Percentage improvement in campaign performance for Scibids compared to the control group.
    - fees_usd (FLOAT): Fees associated with the campaign in USD.
    - fees_value (FLOAT): Value of the fees applied to the campaign, in percentage.
    - roi (FLOAT): Return on investment (ROI) of the campaign.
    - is_ab_test (BOOLEAN): Indicates whether the campaign is part of an A/B test (True/False).
    - start_date_scibids (DATE): Start date of the Scibids campaign.
    - end_date_scibids (DATE): End date of the Scibids campaign.
    - start_date_control (DATE): Start date of the control group.
    - end_date_control (DATE): End date of the control group.
    - scibids_budget_portion (FLOAT): Portion of the budget allocated to Scibids.
    """,

    "new_benchmark": """
    Table: new_benchmark
    Description: Overview table of performance uplift for each unique DSP/KPI combination.

    Columns:
    - dsp (STRING): Advertising platform used (e.g., DV360).
    - kpi (STRING): Key performance indicator of the campaign (e.g., CTR, CPA, CPM, VTR).
    - region (STRING): Geographic region associated with the campaign (may contain null values).
    - weighted_uplift (FLOAT): Weighted performance improvement of the campaign, measured by the relative change in the selected KPI.
    - elligible_ios (INT): Number of insertion orders (IOs) eligible for uplift analysis.
    - not_elligible_ios (INT): Number of IOs not eligible for uplift analysis.
    - elligible_ab_test (INT): Number of campaigns included in an A/B test.
    - not_elligible_ab_test (INT): Number of campaigns excluded from an A/B test.
    """,

    "t_daily_io_features": """
    Table: t_daily_io_features
    Description: Contains information about activated features for each IO per day, 
    as well as the number of alerts. Used to analyze the impact of daily constraint changes 
    on IO performance.

    Columns:
    - id_dsp_io_day_utc (INT): Unique identifier for the DSP IO day in UTC.
    - day_utc (DATE): Corresponding date in UTC.
    - dsp (STRING): Advertising platform used (e.g., DV360, MediaMath).
    - sub_dsp (STRING): Sub-platform or specific category within the DSP.
    - group_object_field_id (STRING): Identifier for the campaign group.
    - kpi_to_optimize (STRING): Main KPI used for optimization.
    - kpi_target (FLOAT): Numerical target for the optimized KPI.
    - min_cpm (FLOAT): Minimum cost per thousand impressions allowed.
    - max_cpm (FLOAT): Maximum cost per thousand impressions allowed.
    - min_margin (FLOAT): Minimum margin allowed for the campaign.
    - min_viz (FLOAT): Minimum required visibility level.
    - remove_min_viz (BOOLEAN): Indicates whether the minimum visibility criterion should be ignored.
    - overwrite_frequency (BOOLEAN): If true, forces the rewriting of frequency rules.
    - log_day (DATE): Data logging date.
    - overwrite_creative_selection (BOOLEAN): If true, forces manual selection of ad creatives.
    - overwrite_li_budget (BOOLEAN): Indicates whether the budget for the insertion order should be forced.
    - force_pacing_asap_li (BOOLEAN): Enables immediate pacing on the insertion order.
    - keep_trusted_inventory (BOOLEAN): If true, retains only trusted inventories.
    - remove_budget_strat_imp (BOOLEAN): Removes certain strategic budget constraints.
    - force_pacing_asap_imp (BOOLEAN): Immediately applies pacing to the impression.
    - use_custom_algo (BOOLEAN): Activates the use of a custom optimization algorithm.
    """,

    "t_campaign_performance_day": """
    Table: t_campaign_performance_day
    Description: Contains performance data for advertising campaigns, 
    including media cost spent, number of impressions, clicks, and other key indicators.

    Columns:
    - client_id (STRING): Unique identifier for the client associated with the campaign.
    - dsp (STRING): Name of the advertising platform (e.g., DV360, Xandr, MediaMath).
    - group_object_field_id (STRING): Identifier for grouped advertising objects.
    - object_field_id (STRING): Unique identifier for the advertising object.
    - gof_flight_id (INT): Flight ID for the group object.
    - of_flight_id (INT): Flight ID for a specific object.
    - billing_scibids_activity (STRING): Defines whether Scibids was active on the campaign at a given time.
    - day_tz (DATE): Date with local timezone.
    - day_utc (DATE): Date in UTC.
    - impressions (INT): Total number of ad impressions.
    - media_cost_currency (FLOAT): Media cost in local currency.
    - media_cost_usd (FLOAT): Media cost converted to USD.
    - revenue_currency (FLOAT): Revenue generated in local currency.
    - revenue_usd (FLOAT): Revenue generated in USD.
    - clicks (INT): Number of recorded clicks.
    - impressions_viewed (FLOAT): Number of impressions actually viewed.
    - impressions_viewed_measured (FLOAT): Number of measured impressions.
    - completed_view (INT): Number of completed views.
    - audible_viewable_completed (INT): Number of completed views with audio enabled.
    - profit_currency (FLOAT): Profit generated in local currency.
    - profit_usd (FLOAT): Profit generated in USD.
    - total_cost_currency (FLOAT): Total cost in local currency.
    - total_cost_usd (FLOAT): Total cost in USD.
    - trueview_views (INT): Number of TrueView views.
    - youtube_conversions (INT): Number of YouTube conversions.
    """,

    "t_flights": """
    Table: t_flights
    Description: Contains tracking information for an IO/OF, including spending objectives and amounts already committed.

    Columns:
    - id_dsp_li_flight (INT): Unique identifier for the DSP flight.
    - dsp (STRING): Advertising platform used.
    - sub_dsp (STRING): Sub-platform used.
    - group_object_field_id (STRING): Group object identifier.
    - object_field_id (STRING): Object identifier.
    - deleted (BOOLEAN): Indicates whether the flight has been deleted.
    - gof_flight_id (INT): GOF flight identifier.
    - of_flight_id (INT): OF flight identifier.
    - flight_billing_activated (BOOLEAN): Indicates whether Scibids was active on this flight.
    - flight_activity (STRING): Flight status (completed, ongoing, or future).
    - gof_start_date_of_tz (STRING): GOF start date (timezone).
    - gof_end_date_of_tz (STRING): GOF end date (timezone).
    - gof_flight_budget_money (FLOAT): Monetary budget for the GOF flight.
    - of_start_date_of_tz (STRING): OF start date (timezone).
    - of_end_date_of_tz (STRING): OF end date (timezone).
    """,

    "t_fit_scores_daily": """
    Table: t_fit_scores_daily
    Description: Table containing the fit score for each day and each IO where we are connected to a client. 
    This score is calculated by the Math Code team to predict our performance on a particular IO.

    Columns:
    - id_dsp_io_day (INT): Unique identifier of the IO per day.
    - dsp (STRING): Advertising platform used.
    - sub_dsp (STRING): Sub-platform used.
    - day (DATE): Date of the fit score.
    - group_object_field_id (STRING): Identifier of the group of objects.
    - final_score (STRING): Final calculated score.
    """,

    "t_insertion_orders": """
    Table: t_insertion_orders
    Description: Table containing information about insertion orders (IO), including advertiser details, budget, KPIs, and optimization parameters.

    Columns:
    - dsp (STRING): Name of the demand-side platform (DSP).
    - sub_dsp (STRING): Identifier of the sub-DSP.
    - parent_object_field_id (INT): Identifier of the parent object.
    - group_object_field_id (INT): Identifier of the group of objects.
    - member_id (INT): Unique identifier of the associated member.
    - advertiser_id (INT): Unique identifier of the advertiser.
    - advertiser_name (STRING): Name of the advertiser.
    - timezone (STRING): Associated timezone.
    - currency_code (STRING): Code of the currency used.
    - exchange_rate (FLOAT): Applied exchange rate.
    - surcouche_setup (BOOLEAN): Indicates if an overlay is activated.
    - keystone_status (STRING): Describes Scibids activity status on the campaign.
    - addressability (STRING): Determines if the IO is optimizable (addressable) or not.
    - ab_test_start_date (DATE): Start date of the A/B test.
    - ab_test_end_date (DATE): End date of the A/B test.
    """,
    
    "t_reach_performance": """
    Table: t_reach_performance
    Description: Table containing information about unique reach performance, including the number of impressions 
    and unique users reached per day and DSP.

    Columns:
    - id_dsp_io_day (INT): Unique identifier of the row per day and DSP.
    - dsp (STRING): Name of the demand-side platform (DSP).
    - insertion_date (DATETIME): Date and time of data insertion.
    - client_id (INT): Unique identifier of the client.
    - start_date (DATETIME): Start date of the campaign.
    - end_date (DATETIME): End date of the campaign.
    - group_object_field_id (INT): Unique identifier of the group of objects.
    - unique_reach_total (INT): Total number of unique users reached.
    - impressions (INT): Total number of recorded impressions.
    - prev_value (INT): Previous value of unique reach.
    - last_new_unique_reach (INT): Latest newly reached unique value.
    - last_new_reach_value (BOOLEAN): Indicator if the latest reach value is valid.
    """,

    "t_pixel_performance": """
    Table: t_pixel_performance
    Description: Table containing the performance of conversion pixels, tracking conversion events per hour and DSP.

    Columns:
    - dsp (STRING): Name of the demand-side platform (DSP).
    - sub_dsp (STRING): Identifier of the sub-DSP.
    - client_id (INT): Unique identifier of the client.
    - group_object_field_id (INT): Unique identifier of the group of objects.
    - conversion_pixel_id (INT): Identifier of the conversion pixel.
    - conversion_pixel_name (STRING): Name of the conversion pixel.
    - object_field_id (INT): Identifier of the associated object field.
    - day_tz (DATETIME): Date of the day with local timezone.
    - day_utc (DATETIME): Date of the day in UTC.
    - date_hour_tz (DATETIME): Hour of the event with local timezone.
    - date_hour_utc (DATETIME): Hour of the event in UTC.
    - pixel_type (INT): Type of pixel used.
    - billing_scibids_activity (STRING): Indicates whether Scibids was active at the time of conversion.
    - pixel_origin (STRING): Origin of the pixel.
    - pixel_post_click_convs (FLOAT): Number of post-click conversions.
    - pixel_post_view_convs (FLOAT): Number of post-view conversions.
    - pixel_total_convs (FLOAT): Total number of conversions.
    """
}


OPENAI_API_KEY = "****" # Replace with your OpenAI API key

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Set Google Cloud project ID
PROJECT_ID = "capstone-448012"

# Initialize BigQuery Client
client_bq = bigquery.Client(project=PROJECT_ID)

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
        # Check query validity (Dry Run)
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = client_bq.query(sql_query, job_config=job_config)
        st.success("✅ Query is valid! Running execution...")

        # Execute query if valid
        query_job = client_bq.query(sql_query)
        results = query_job.result().to_dataframe()

        if results.empty:
            st.warning("⚠️ Query executed successfully but returned no results.")
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
        # Few-shot examples to guide the model clearly
    few_shot_examples = """
    # Example 1:
    User Request: "What is the total amount spent by the client ID_SELECTED?"
    SQL Query:
    SELECT client_id, 
        SUM(revenue_currency) AS spend, 
        SUM(revenue_usd) AS spend_usd
    FROM `t_campaign_performance_day`
    WHERE client_id = 'ID_SELECTED'
    GROUP BY client_id;

    # Example 2:
    User Request: "What are the ongoing A/B tests?"
    SQL Query:
    SELECT group_object_field_id, ab_test_start_date, ab_test_end_date
    FROM `t_insertion_orders`
    WHERE CURRENT_DATE() BETWEEN ab_test_start_date AND ab_test_end_date;
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert SQL assistant specialized in digital advertising and data analytics. "
                "Generate only the SQL query while strictly following the provided database schemas and user request.\n"
                "### Few-Shot Examples:\n"
                f"{few_shot_examples}\n\n"
                "### Keys Identification (for joins and filtering):\n"
                f"{relevant_keys}\n\n"
                "### Database Schema:\n"
                f"{relevant_schemas}"
            ),
        },
        {
            "role": "user",
            "content": f"Instruction:\n{user_query}\n\nGenerate only the SQL query without explanation.",
        },
    ]


    # Call OpenAI API to generate the SQL query
    completion = client.chat.completions.create(
        model="gpt-4o",  # High-performance model for SQL generation
        messages=messages,
        temperature=0.2,  # Low temperature for deterministic output
    )

    return completion.choices[0].message.content
