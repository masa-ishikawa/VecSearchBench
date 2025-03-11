# Vector Search TPS Measurement Tool

This repository provides a simple tool for measuring the Transactions Per Second (TPS) of vector search operations. It is designed with an extensible architecture that enables you to measure TPS for any vector store by extending the provided abstract class.

## Overview

- **Simple Vector Search TPS Measurement**:  
  This tool is specifically built to measure the TPS of vector search operations.

- **Extensible Framework**:  
  By extending the `AbstractLoaderTest` class, you can implement TPS measurement for any vector store. All you need to do is implement the abstract method `execute_query` to perform your specific vector search.

- **Customizable Test Parameters**:  
  The `run_test` method of `AbstractLoaderTest` allows you to set arbitrary values for TPS, duration, and timeout, providing flexibility for various testing scenarios.

- **Sample Implementation**:  
  The `OCI_Postgres_LoadTest` class is provided as a sample concrete implementation of `AbstractLoaderTest`. It demonstrates how to implement the `execute_query` method for a PostgreSQL-based vector search measurement. Use this sample as a reference for your own implementations.

## How It Works

1. **Implement `execute_query`**:  
   To measure the TPS of your vector search, create a subclass of `AbstractLoaderTest` and implement the `execute_query` method. This method should execute the vector search operation you wish to measure.

2. **Run the Test**:  
   Call the `run_test` method on your instance after implementing `execute_query`. This method will:
   - Execute the specified number of operations per second (TPS).
   - Run for the defined duration.
   - Use the provided timeout value for each batch of operations.
   - Log the results, including total operations, successes, failures, and QPS (queries per second).

3. **Review the Results**:  
   After the test, results are logged to a daily log file with details such as the total number of operations, number of successful operations, number of errors, and the calculated QPS.
