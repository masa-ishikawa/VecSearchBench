import logging
import os
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
        if config is None:
            AbstractLoaderTest.setup_logging()
            logging.getLogger().setLevel(logging.DEBUG)
            config = self.load_config("config.yaml")
        self.config = config
        super().__init__(tps, duration, timeout)

        # --- DEBUG LOG for Milvus ---
        logging.debug("Connecting to Milvus with:")
        logging.debug(f"  uri             : {self.config['milvus_uri']}")
        logging.debug(f"  collection_name : {self.config['collection_name']}")
        logging.debug(f"  output_fields   : {self.config['output_fields']}")

        self.client = MilvusClient(uri=self.config["milvus_uri"])

        # --- DEBUG LOG for GenAI ---
        logging.debug("Embedding using OCI GenAI with:")
        logging.debug(f"  endpoint        : {self.config['service_endpoint']}")
        logging.debug(f"  compartment_id  : {self.config['compartment_id']}")
        logging.debug(f"  model_id        : {self.config['emb_llm_id']}")
        logging.debug(f"  query           : {query}")

        genai_client = GenerativeAiInferenceClient(
            config={},
            signer=InstancePrincipalsSecurityTokenSigner(),
            service_endpoint=self.config["service_endpoint"]
        )
        res = genai_client.embed_text(
            embed_text_details=EmbedTextDetails(
                inputs=[query],
                serving_mode=OnDemandServingMode(model_id=self.config["emb_llm_id"]),
                compartment_id=self.config["compartment_id"],
                input_type="SEARCH_QUERY"
            )
        )
        self.config["embedding_vector"] = res.data.embeddings

    @staticmethod
    def load_config(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    
    @property
    def EMBEDDING_VECTOR(self):
        return self.config["embedding_vector"]

    def execute_query(self):
        try:
            result = self.client.search(
                collection_name=self.config["collection_name"],
                data=self.EMBEDDING_VECTOR,
                output_fields=self.config["output_fields"]
            )
            return result
        except Exception as e:
            logging.error(f"Error during query execution: {e}")
            raise

if __name__ == "__main__":
    tps = int(os.environ.get("TPS", 10))
    duration = int(os.environ.get("DURATION", 3))
    timeout = int(os.environ.get("TIMEOUT", 10))
    query = os.environ.get("WORD", "サヴォワ地方はどこの国？")

    tester = MilvusLoadTest(
        tps=tps,
        duration=duration,
        timeout=timeout,
        query=query
    )
    tester.run_test()
