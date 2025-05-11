import logging
import yaml
from abstract_loader_test import AbstractLoaderTest
from pymilvus import MilvusClient
from oci.auth.signers.instance_principals_security_token_signer import InstancePrincipalsSecurityTokenSigner
from oci.generative_ai_inference.generative_ai_inference_client import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode
)

class MilvusLoadTest(AbstractLoaderTest):
    def __init__(self, tps, duration, timeout, query, config=None):
        """
        :param tps: Number of SQL queries per second
        :param duration: Test execution time in seconds
        :param timeout: Timeout value (in seconds) for each batch
        :param config: Configuration content. If not provided, "config.yaml" will be loaded and logging will be configured automatically.
        """
        if config is None:
            AbstractLoaderTest.setup_logging()
            config = self.load_config("config.yaml")
        self.config = config
        super().__init__(tps, duration, timeout)
        # Create Milvus connection
        self.client = MilvusClient(
            uri=self.config["milvus_uri"]
        )
        genai_client = GenerativeAiInferenceClient(
            config={},
            signer=InstancePrincipalsSecurityTokenSigner(),
            service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"
        )
        res = genai_client.embed_text(
            embed_text_details=EmbedTextDetails(
                inputs=[query],
                serving_mode=OnDemandServingMode(
                    model_id="cohere.embed-multilingual-v3.0",
                ),
                compartment_id=self.config["compartment_id"],
                input_type="SEARCH_QUERY"
            )
        )
        self.config["embedding_vector"] = res.data.embeddings

    @staticmethod
    def load_config(file_path):
        """Load the configuration file (YAML)"""
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    
    @property
    def EMBEDDING_VECTOR(self):
        """The embedding vector value to be passed to the SQL query."""
        return self.config["embedding_vector"]

    def execute_query(self):
        """
        Executes the query and returns the result.
        Appends a unique string to bypass cache.
        """
        try:
            result = self.client.search(
                collection_name=self.config["collection_name"],
                data=self.EMBEDDING_VECTOR,
                output_fields=self.config["output_fields"]
            )
            return result
        except Exception as e:
            logging.error(f"Error during query execution: {e}")
            raise e

if __name__ == "__main__":
    # Main execution: create an instance with the required parameters and run the test
    tester = MilvusLoadTest(tps=100, duration=10, timeout=3, query="サヴォワ地方はどこの国？")
    tester.run_test()
