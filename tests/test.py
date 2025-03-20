"""
Unit tests for the `sql_query_and_plot_generator` package.

This module contains test cases for verifying the correctness of 
SQL query generation, execution, and utility functions.
"""

import unittest
from sql_query_and_plot_generator.main import main
from sql_query_and_plot_generator.utils import clean_sql_query, execute_bigquery, generate_sql, recognize_tables


class TestMain(unittest.TestCase):
    """
    Unit tests for the `main` function in `main.py`.

    This class ensures that the Streamlit-based UI runs properly and does not throw errors.
    """

    def test_main_function(self):
        """Test if the `main` function runs without errors."""
        try:
            main()
            self.assertTrue(True)  # Test passes if no exception occurs
        except Exception as e:
            self.fail(f"main() raised an exception: {e}")


class TestUtils(unittest.TestCase):
    """
    Unit tests for utility functions in `utils.py`.

    This class tests SQL query handling, execution, and table recognition functions.
    """

    def test_clean_sql_query(self):
        """Test if `clean_sql_query` correctly cleans SQL queries."""
        raw_sql = "SELECT * FROM users WHERE age > 30;"
        cleaned_sql = clean_sql_query(raw_sql)
        self.assertIn("users", cleaned_sql)

    def test_recognize_tables(self):
        """Test if `recognize_tables` correctly identifies table names."""
        user_query = "Find all campaigns with high engagement."
        tables = recognize_tables(user_query)
        self.assertIsInstance(tables, list)

    def test_generate_sql(self):
        """Test SQL query generation from user query and selected tables."""
        user_query = "Show me all clients with revenue over 1000."
        selected_tables = ["t_campaign_performance_day"]
        sql_query = generate_sql(user_query, selected_tables)
        self.assertIn("SELECT", sql_query.upper())

    def test_execute_bigquery(self):
        """Test if `execute_bigquery` runs without errors (mock test)."""
        test_sql = "SELECT 1 AS test_column"
        try:
            result = execute_bigquery(test_sql)  # This may need a mock
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"execute_bigquery raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
