# Domains


## Domain 1: Data Preparation for Machine Learning (ML)

### 1.1: Ingest and store data.

1. Ingest Data
   
Data ingestion in AWS refers to the process of collecting and transferring data from various sources into storage systems, where it can be accessed for machine learning. AWS provides multiple services for this purpose, catering to batch, real-time, and event-driven use cases.

    1. Batch Ingestion:
Batch ingestion is suitable for collecting data at scheduled intervals or in large, bulk volumes.

Amazon S3 Transfer Acceleration: Allows fast, secure file transfers into Amazon S3 from remote locations. It's useful for large-scale datasets like logs, images, and videos used for training ML models.

AWS DataSync: Automates the transfer of data between on-premises storage and AWS, ideal for migrating existing datasets into the cloud for machine learning.

AWS Snowball / Snowmobile: If the data is very large (in terabytes or petabytes), these physical devices allow secure transfer of large datasets to AWS. After ingesting the data, it can be uploaded to S3 for further processing.

AWS Glue: AWS Glue can ingest data from a variety of sources, including databases, data lakes, and file systems, transform the data, and load it into a target data store such as S3 or Redshift for ML.

    2. Real-time Ingestion:
Real-time data ingestion is essential for use cases requiring immediate insights, like IoT data processing or real-time user interactions.

Amazon Kinesis Data Streams: Allows ingestion of real-time streaming data, such as log data, application event data, or sensor data, into AWS. This data can then be processed for ML use cases like anomaly detection or real-time recommendations.

Amazon Kinesis Data Firehose: Automatically ingests streaming data and stores it in Amazon S3, Redshift, or Elasticsearch, making it easy to process and store real-time data.

AWS IoT Core: For IoT-specific data ingestion, AWS IoT Core helps manage, ingest, and process data from IoT devices. This is useful for ML models that need real-time telemetry data.

Amazon Managed Streaming for Apache Kafka (MSK): Apache Kafka is another option for handling real-time event streams. With MSK, AWS offers a fully managed Kafka service for real-time data ingestion for event-driven architectures and machine learning.

    3. Event-driven Ingestion:
Event-driven architectures are suitable for triggering data ingestion when specific events occur, rather than continuous or scheduled ingestion.

Amazon Simple Notification Service (SNS) and Amazon Simple Queue Service (SQS): These services are often used to ingest data based on events. For instance, an application might generate a message when a user uploads a new image, and this triggers an ingestion pipeline for further processing by an ML model.

AWS Lambda: Serverless compute that can be triggered by events (e.g., file uploads to S3, messages in SQS). It can preprocess and ingest data into storage systems for further use in ML workflows.

2. Store Data
Once the data is ingested, it needs to be stored in a way that makes it accessible for ML models, training, and inference. AWS offers a variety of storage services optimized for different types of data—structured, unstructured, streaming, and batch.

* Amazon S3 (Simple Storage Service):
Use Case: Most commonly used for storing large datasets, especially unstructured data like images, videos, logs, etc.
Machine Learning Integration: S3 integrates directly with machine learning services like Amazon SageMaker, making it an ideal choice for storing training data and models.
Features:
Cost-effective and scalable storage for large volumes of data.
Supports versioning, encryption, and lifecycle management.
S3 Select allows you to retrieve specific parts of an object for ML purposes without loading the entire dataset.
Data Lake capabilities for analytics and ML pipelines.
* Amazon RDS (Relational Database Service):
Use Case: For structured data like transactional records, time-series data, or metadata required for ML tasks.
Machine Learning Integration: RDS can be used with services like SageMaker to store and query structured data for training.
Features: Supports multiple database engines (e.g., MySQL, PostgreSQL, Oracle), highly available, and automated backups.
* Amazon Redshift:
Use Case: For large-scale, structured data warehouses. Redshift is ideal for storing and querying vast amounts of structured data for ML applications like predictive analytics or business intelligence.
Machine Learning Integration: Redshift ML allows you to build, train, and deploy machine learning models using SQL queries, integrating directly with SageMaker.
Features:
Fast query performance using SQL and machine learning.
Native integration with S3 for data lake use cases.
Redshift Spectrum for querying data stored in S3 without having to load it into Redshift.
* Amazon DynamoDB:
Use Case: For real-time, low-latency, and high-availability needs, especially for unstructured data. It’s commonly used for IoT, gaming, or session data.
Machine Learning Integration: Directly integrated with SageMaker for inferencing, and suitable for scenarios where real-time access to training data or inferences is necessary.
Features: NoSQL database, supports key-value and document data structures, and can scale automatically.
* Amazon Elastic File System (EFS):
Use Case: Suitable for storing shared file systems that need to be accessed by multiple instances or services, like training datasets for distributed ML training.
Machine Learning Integration: SageMaker can use EFS to access data for distributed training jobs.
Features: Scalable, fully managed file system for Linux workloads.
* Amazon S3 Glacier:
Use Case: For storing archived data, such as historical datasets, or backups that may need to be accessed later for ML purposes.
Features: Cost-effective long-term storage, with retrieval options ranging from minutes to hours. Ideal for datasets that are infrequently accessed but still necessary for model retraining or compliance.
* Amazon FSx for Lustre:
Use Case: Optimized for high-performance computing (HPC) and machine learning workloads where fast access to large datasets is necessary.
Machine Learning Integration: Often used for large-scale ML training jobs that require fast, parallel processing, and it integrates with Amazon S3 for data storage.
Features: High throughput, low latency, and designed for intensive ML workloads.

### 1.2: Transform data and perform feature engineering.

### 1.3: Ensure data integrity and prepare data for modeling.


## Domain 2: ML Model Development

### 2.1: Choose a modeling approach.

### 2.2: Train and refine models.

### 2.3: Analyze model performance.


## Domain 3: Deployment and Orchestration of ML Workflows

### 3.1: Select deployment infrastructure based on existing architecture and requirements.

### 3.2: Create and script infrastructure based on existing architecture and requirements.

### 3.3: Use automated orchestration tools to set up continuous integration and continuous delivery (CI/CD) pipelines.


## Domain 4: ML Solution Monitoring, Maintenance, and Security

### 4.1: Monitor model inference.

### 4.2: Monitor and optimize infrastructure and costs.

### 4.3: Secure AWS resources.