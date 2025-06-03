import os
import yaml
import psycopg2
import logging
from dotenv import load_dotenv
from psycopg2 import pool
from abstract_loader_test import AbstractLoaderTest
from oci.config import from_file
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode
import random, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class OCI_Postgres_LoadTest(AbstractLoaderTest):
    def __init__(self, tps, duration, timeout, word, config=None):
        if config is None:
            AbstractLoaderTest.setup_logging()
            config = self.load_config('config.yaml')
        self.config = config
        self.word = word
        self.embedding_vector = self.embed_word(word)
        super().__init__(tps, duration, timeout)

        # PostgreSQL connection pool
        logging.debug("Creating PostgreSQL connection pool with the following parameters:")
        logging.debug(f"  host     : {self.config['pgvector_dbhost']}")
        logging.debug(f"  port     : 5432")
        logging.debug(f"  dbname   : {self.config['pgvector_dbname']}")
        logging.debug(f"  user     : {self.config['pgvector_username']}")
        logging.debug(f"  password : {'*' * len(self.config['pgvector_password'])}")

        self.conn_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=self.tps * 2,
            host=self.config['pgvector_dbhost'],
            port=5432,
            dbname=self.config['pgvector_dbname'],
            user=self.config['pgvector_username'],
            password=self.config['pgvector_password']
        )

    def embed_word(self, word):
        oci_config = from_file()
        endpoint = self.config["service_endpoint"]
        compartment_id = self.config["compartment_id"]
        model_id = self.config.get("emb_llm_id", "cohere.embed-multilingual-v3.0")

        client = GenerativeAiInferenceClient(config=oci_config, service_endpoint=endpoint)
        result = client.embed_text(
            embed_text_details=EmbedTextDetails(
                inputs=[word],
                serving_mode=OnDemandServingMode(model_id=model_id),
                compartment_id=compartment_id,
                is_echo=True,
                truncate="NONE"
            )
        )
        return result.data.embeddings[0]

    @staticmethod
    def load_config(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    @property
    def BASE_SQL(self):
        return f"""
        SELECT w.text 
        FROM WIKI_JA_EMBEDDINGS_20250401_HNSW w
        ORDER BY embedding <=> %s::vector
        LIMIT 4
        """

    def get_connection(self):
        conn = self.conn_pool.getconn()
        # if not hasattr(conn, '_ef_search_set'):
        #     with conn.cursor() as cur:
        #         cur.execute("SET hnsw.ef_search = 10;")
        #         logging.info("SET hnsw.ef_search = 10 (once per connection);")
        #     conn._ef_search_set = True  # フラグを立てる
        return conn

    def put_connection(self, conn):
        self.conn_pool.putconn(conn)

    def close_all(self):
        self.conn_pool.closeall()

    def execute_query(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # dummy_value = random.randint(1, 10000)
                sql = self.BASE_SQL
                def run_sql():
                    start_time = time.time()
                    cur.execute(sql, (self.embedding_vector,))
                    result = cur.fetchall()
                    elapsed_ms = (time.time() - start_time) * 1000
                    # print(f"Query execution + fetch time: {elapsed_ms:.2f} ms")
                    return result

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_sql)
                    return future.result(timeout=self.timeout)

        except TimeoutError:
            logging.error(f"Query timed out after {self.timeout} seconds.")
            raise
        except Exception as e:
            logging.error("Error during query execution: %s", e)
            raise
        finally:
            self.put_connection(conn)

if __name__ == '__main__':
    tps = int(os.environ.get("TPS",10))
    duration = int(os.environ.get("DURATION", 3))
    timeout = int(os.environ.get("TIMEOUT", 10))
    word = os.environ.get("WORD", "サヴォワ地方はどこの国？")

    tester = OCI_Postgres_LoadTest(
        tps=tps,
        duration=duration,
        timeout=timeout,
        word=word
    )
    tester.run_test()
