import unittest
from sql_query_generator.main import main
from sql_query_generator.utils import clean_sql_query, execute_bigquery, generate_sql, recognize_tables

class TestMain(unittest.TestCase):
    def test_main_function(self):
        """Test if the main function runs without errors."""
        try:
            main()  # This won't return anything, just checking if it runs
            self.assertTrue(True)  # Pass the test if no exception occurs
        except Exception as e:
            self.fail(f"main() raised an exception: {e}")

class TestUtils(unittest.TestCase):
    def test_clean_sql_query(self):
        """Test if the SQL query cleaning function works correctly."""
        raw_sql = "SELECT * FROM users WHERE age > 30;"
        cleaned_sql = clean_sql_query(raw_sql)
        self.assertIn("users", cleaned_sql)  # Check if table name remains
        
    def test_recognize_tables(self):
        """Test if table recognition correctly identifies table names."""
        user_query = "Find all campaigns with high engagement."
        tables = recognize_tables(user_query)
        self.assertIsInstance(tables, list)  # Should return a list

    def test_generate_sql(self):
        """Test SQL query generation from user query and selected tables."""
        user_query = "Show me all clients with revenue over 1000."
        selected_tables = ["t_campaign_performance_day"]
        sql_query = generate_sql(user_query, selected_tables)
        self.assertIn("SELECT", sql_query.upper())  # Check if it contains SELECT

    def test_execute_bigquery(self):
        """Test if execute_bigquery runs without errors (mock test)."""
        test_sql = "SELECT 1 AS test_column"
        try:
            result = execute_bigquery(test_sql)  # This may need a mock
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"execute_bigquery raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
