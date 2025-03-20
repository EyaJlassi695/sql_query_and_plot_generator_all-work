import re

import pandas as pd
import streamlit as st
from utils import clean_sql_query, execute_bigquery, generate_sql, recognize_tables

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

# Streamlit UI
st.set_page_config(page_title="SQL Query Generator", layout="wide")

# Add Scibids logo at the top
st.image("Images\DV-SCIBIDS-650.png", width=250) 

st.title("🛠️ SQL Query Generator")
st.write("Enter a natural language request, and the system will generate the corresponding SQL query.")

# Layout with two columns
col1, col2 = st.columns([1, 2])

# Display section
with col1:
    st.markdown("## 📊 Table Schema")

    # Dropdown to select a table dynamically
    selected_table = st.selectbox("Select a table to view its schema:", options=list(DB_SCHEMAS.keys()))

    # Extract schema details
    schema_details = DB_SCHEMAS[selected_table]
    
    # Extract table name
    schema_lines = schema_details.split("\n")
    table_name = schema_lines[0].replace("Table: ", "").strip() if "Table: " in schema_lines[0] else selected_table

    # Extract column details (Name & Description only)
    column_data = []
    for line in schema_lines:
        if "- " in line and ":" in line:
            parts = line.replace("- ", "").split(":")
            column_name = parts[0].strip()
            
            # Extract description (ignoring type)
            column_desc = ":".join(parts[1:]).strip()

            column_data.append({"Column Name": column_name, "Description": column_desc})

    # Convert to DataFrame for clean display
    df_schema = pd.DataFrame(column_data)

    # Use an expander to hide/show schema details
    with st.expander(f"🔍 **View Schema for `{table_name}`**", expanded=True):
        st.markdown(f"### 📝 Table Name: `{table_name}`", unsafe_allow_html=True)
        st.table(df_schema[["Column Name", "Description"]])

# Query generation section
with col2:
    st.subheader("✏️ Enter your request")
    user_input = st.text_area("Describe your SQL query:", placeholder="Example: Get the average weighted uplift for each DSP and KPI.")

    if st.button("Generate & Run SQL"):
        if user_input.strip():
            selected_tables = recognize_tables(user_input)
            st.subheader("📌 Identified Tables:")
            st.write(", ".join(selected_tables))

            if selected_tables:
                raw_sql_query = generate_sql(user_input, selected_tables)  # Generate query
                cleaned_sql_query = clean_sql_query(raw_sql_query)  # Clean query

                st.subheader("📝 SQL Query:")
                st.code(cleaned_sql_query, language="sql")

                # Run query on BigQuery
                st.subheader("📊 Query Results:")
                results = execute_bigquery(cleaned_sql_query)
                if results is not None :
                    st.table(results)

            else:
                st.warning("No relevant tables were identified.")
        else:
            st.warning("Please enter a valid query description.")
