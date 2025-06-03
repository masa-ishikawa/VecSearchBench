import time
import threading
import logging
import random
import string
import datetime
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, wait


class AbstractLoaderTest(ABC):
    def __init__(self, tps, duration, timeout):
        """
        :param tps: Number of operations per second
        :param duration: Test execution time in seconds
        :param timeout: Timeout value (in seconds) for each batch
        """
        self.tps = tps
        self.duration = duration
        self.timeout = timeout
        self.total_queries = 0
        self.success_count = 0
        self.error_count = 0
        self.lock = threading.Lock()

    @abstractmethod
    def execute_query(self):
        """
        Abstract method to execute the test operation.
        This method can be overridden for non-database purposes as well.
        """
        pass

    def run_test(self):
        """
        Executes tps operations per second (using execute_query) and evaluates the results within the timeout period.
        """
        with ThreadPoolExecutor(max_workers=self.tps * 2) as executor:
            for i in range(self.duration):
                batch_futures = [executor.submit(self.execute_query) for _ in range(self.tps)]
                batch_start = time.time()
                done, not_done = wait(batch_futures, timeout=self.timeout)

                for future in done:
                    try:
                        future.result()
                        with self.lock:
                            self.success_count += 1
                    except Exception:
                        with self.lock:
                            self.error_count += 1

                for future in not_done:
                    with self.lock:
                        self.error_count += 1
                    future.cancel()

                batch_elapsed = time.time() - batch_start
                if batch_elapsed < 1.0:
                    time.sleep(1.0 - batch_elapsed)
                with self.lock:
                    self.total_queries += self.tps

        self.report_results()

    def report_results(self):
        logging.info("=== Test Results ===")
        logging.info("Total operations: %d", self.total_queries)
        logging.info("Successful operations: %d", self.success_count)
        logging.info("Failed operations: %d", self.error_count)
        qps = self.total_queries / self.duration
        logging.info("QPS: %.2f", qps)

    @staticmethod
    def setup_logging():
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        log_filename = f"vectordbtest_{current_date}.log"
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        root_logger.handlers = [file_handler, console_handler]

    def _generate_unique_comment(self):
        """Generate a random string for cache busting"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))
