import os
import yaml
import logging
import oracledb
import sys
import random
import time

from dotenv import load_dotenv
from abstract_loader_test import AbstractLoaderTest
from oci.config import from_file
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode

class BaseDB_LoadTest(AbstractLoaderTest):
    def __init__(self, tps, duration, timeout, word, config=None):
        if config is None:
            AbstractLoaderTest.setup_logging()
            config = self.load_config('config.yaml')
        self.config = config
        self.word = word
        self.embedding_vector = self.embed_word(word)
        self.embedding_vector_str = "[" + ",".join([f"{x:.8f}" for x in self.embedding_vector]) + "]"
        super().__init__(tps, duration, timeout)

        print(oracledb.__version__)
        try:
            oracledb.init_oracle_client()
        except oracledb.ProgrammingError:
            pass  # すでに初期化済み

        host = self.config["basedb_host"]
        port = self.config["basedb_port"]
        service_name = self.config["basedb_service_name"]
        self.dsn = f"{host}:{port}/{service_name}"

        logging.debug("Creating Oracle session pool with the following parameters:")
        logging.debug(f"  user         : {self.config['basedb_username']}")
        logging.debug(f"  password     : {'*' * len(self.config['basedb_password'])}")
        logging.debug(f"  host         : {host}")
        logging.debug(f"  port         : {port}")
        logging.debug(f"  service_name : {service_name}")
        logging.debug(f"  dsn          : {self.dsn}")

        self.pool = oracledb.SessionPool(
            user=self.config["basedb_username"],
            password=self.config["basedb_password"],
            dsn=self.dsn,
            min=1,
            max=self.tps * 2,
            increment=1,
            getmode=oracledb.SPOOL_ATTRVAL_WAIT
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

    def generate_sql(self):
        return f"""
            SELECT /*+ VECTOR_INDEX_SCAN(w EMBEDDING_IVF_IDX_20250401) */ w.text
            FROM WIKI_JA_EMBEDDINGS_20250401_IVF w
            ORDER BY VECTOR_DISTANCE(
                    w.embedding,
                    TO_VECTOR(:embedding)
                )
            FETCH APPROX FIRST 4 ROWS ONLY
        """

    # def generate_sql(self):
    #     return f"""
    #         SELECT /*+ VECTOR_INDEX_SCAN( w embedding_hnsw_idx_20250401) */ w.text
    #         FROM WIKI_JA_EMBEDDINGS_20250401_HNSW w
    #         --WHERE :dummy IS NOT NULL
    #         ORDER BY VECTOR_DISTANCE(
    #             w.embedding,
    #             TO_VECTOR(:embedding)
    #         )
    #         FETCH APPROX FIRST 4 ROWS ONLY
    #     """

    def execute_query(self):
        conn = self.pool.acquire()
        try:
            with conn.cursor() as cur:
                sql = self.generate_sql()
                dummy_value = random.randint(1, 10000)

                def run_sql():
                    start_time = time.time()
                    try:
                        cur.execute(sql, {
                            # "dummy": dummy_value,
                            "embedding": self.embedding_vector_str
                        })
                        result = cur.fetchall()
                        elapsed_ms = (time.time() - start_time) * 1000
                        # print(f"Query execution + fetch time: {elapsed_ms:.2f} ms")
                        # print([row[0].read() if isinstance(row[0], oracledb.LOB) else row[0] for row in result])
                        return result
                    except Exception as e:
                        logging.error(f"Error inside run_sql: {e}")
                        raise

                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                with ThreadPoolExecutor(max_workers=1) as executor:
                    # print("Submitting SQL for execution...")
                    future = executor.submit(run_sql)
                    try:
                        return future.result(timeout=self.timeout)
                    except TimeoutError:
                        logging.error(f"Query timed out after {self.timeout} seconds.")
                        raise
        finally:
            self.pool.release(conn)

    def close_all(self):
        self.pool.close()

if __name__ == '__main__':
    tps = int(os.environ.get("TPS", 10))
    duration = int(os.environ.get("DURATION", 3))
    timeout = int(os.environ.get("TIMEOUT", 10))
    word = os.environ.get("WORD", "サヴォワ地方はどこの国？")

    tester = BaseDB_LoadTest(
        tps=tps,
        duration=duration,
        timeout=timeout,
        word=word
    )
    tester.run_test()
