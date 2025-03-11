import yaml
import psycopg2
from psycopg2 import pool
from abstract_loader_test import AbstractLoaderTest
import logging


class OCI_Postgres_LoadTest(AbstractLoaderTest):
    def __init__(self, tps, duration, timeout, config=None):
        """
        :param tps: Number of SQL queries per second
        :param duration: Test execution time in seconds
        :param timeout: Timeout value (in seconds) for each batch
        :param config: Configuration content. If not provided, 'config.yaml' will be loaded and logging will be configured automatically.
        """
        if config is None:
            AbstractLoaderTest.setup_logging()
            config = self.load_config('config.yaml')
        self.config = config
        super().__init__(tps, duration, timeout)
        # Create a PostgreSQL connection pool (minconn=1, maxconn is tps * 2)
        self.conn_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=self.tps * 2,
            host=self.config['dbhost'],
            port=5432,
            dbname=self.config['dbname'],
            user=self.config['username'],
            password=self.config['password']
        )

    @staticmethod
    def load_config(file_path):
        """Load the configuration file (YAML)"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    @property
    def BASE_SQL(self):
        """SQL template string containing the {unique} placeholder."""
        return f"""
SELECT * FROM {self.config['table']} 
ORDER BY embedding <-> %s::vector
LIMIT 1
-- no-cache: {{unique}}
"""

    @property
    def EMBEDDING_VECTOR(self):
        """The embedding vector value to be passed to the SQL query."""
        return self.config['EMBEDDING_VECTOR']

    def get_connection(self):
        return self.conn_pool.getconn()

    def put_connection(self, conn):
        self.conn_pool.putconn(conn)

    def close_all(self):
        self.conn_pool.closeall()

    def execute_query(self):
        """
        Retrieves a connection from the pool, executes the SQL query and returns the result.
        Appends a unique string to bypass cache.
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                # _generate_unique_comment is defined in AbstractLoaderTest
                unique = self._generate_unique_comment()
                sql = self.BASE_SQL.format(unique=unique)
                cur.execute(sql, (self.EMBEDDING_VECTOR,))
                result = cur.fetchall()
            conn.commit()
            return result
        except Exception as e:
            logging.error("Error during query execution: %s", e)
            raise e
        finally:
            if conn:
                self.put_connection(conn)


if __name__ == '__main__':
    # Main execution: create an instance with the required parameters and run the test
    tester = OCI_Postgres_LoadTest(tps=100, duration=10, timeout=3)
    tester.run_test()
