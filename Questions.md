# Sample Questions

## Collect, Ingest, and Store Data

1. A retail company wants to use an Amazon Elastic File System (Amazon EFS) file server for their machine learning (ML) workload. The Amazon EFS file server will be used to store data for model training that will be accessed from a fleet of training instances.
Which AWS service would best be used for efficiently extracting a large dataset from a file system hosted on Amazon EC2 to an Amazon EFS file server?
  
    *AWS DataSync is designed for efficient transferring of large amounts of data between on-premises storage and AWS, or between AWS storage services.*

2. A data science team has been tasked with building a machine learning (ML) model for detecting disease outbreaks. The data will be trained on a large dataset of medical records, including lab results, treatment histories, medication data, and more. The team needs to decide on a storage solution for hosting and training the ML model.  
Which storage services are the best choices for this project? (Select TWO.)

    *Amazon S3 and Amazon EFS would be the best choices for this solution. Amazon S3 would be the best choice for storing the initial dataset and for copying and loading the data to Amazon EFS. Amazon EFS would serve as the storage for model training because the file system provides distributed and concurrent access for higher performance.*


1. A machine learning (ML) workload has the following requirements: shared storage to train a machine learning model simultaneously on a massive amount of storage, extremely low latency, and high throughput.  
Which storage service would be the most effective choice?  
  
    *Amazon FSx for Lustre provides high performance and concurrent access to a file system that is suitable for ML training and requires the highest performance requirements.*

1. Raw click-stream data has been ingested into a centralized data store that will ultimately be used for training a machine learning (ML) algorithm to personalize recommendations. The raw data consists of user IDs, time stamps, session duration, geolocation, and more.  
Which data format should the data be transformed to for efficient storing and processing?
  
    *Parquet provides a columnar data structure that is efficient for storing click-stream data. Parquet's columnar storage and compression makes it a good choice for machine learning.*

1. You are working on a machine learning (ML) project that requires ingesting and processing large volumes of data from various sources. As the data is ingested with Amazon Kinesis Data Streams and stored in Amazon S3, you have been experiencing performance issues. High latency, slow data transfer, and capacity limitations have all been occurring.  
How could you mitigate these issues?
  
    *Performance issues can occur with high amounts of data being sent to a single storage destination (Amazon S3). This can lead to latency and slow data transfer. Compressing data prior to sending it to Amazon S3 and using Amazon S3 multi-part uploads can reduce the bandwidth requirements and speed up data transfer times. Using dynamic partitioning with Amazon Data Firehose can distribute data load and alleviate capacity issues of sending data to a single storage location.*


6. A recommendation model during training needs access to a redundant and highly available data store. It must securely store images and serve the images during training.  
Which of the AWS storage options best meets these requirements?
  
    *Amazon S3 provides durable object storage with high availability. Amazon S3 is well-suited for read-only data, like training data.*

7. A data scientist at a financial institution is in the early stages of the machine learning (ML) lifecycle. They are deciding which data to collect for an ML algorithm to predict loan defaults.  
Which dataset should the data scientist exclude due to poor data quality?
  
    *A dataset that consists of only loan applicants who currently hold a loan with the institution (The dataset is non-representative and does not reflect the overall portion of applicants that are applying for loans.)*

8. You are a member of a machine learning (ML) team that is tasked with building a real-time product recommendation engine for an e-commerce website. The data used for recommendations will consist of unstructured data, such as purchases, browsing history, customer details, and more. The team needs to decide on a file format that provides efficient parsing and analysis of the dataset as it is streamed in real time.  
Which file format should the team use?
  
    *When using JSON Lines, there are separate JSON objects for each line, which helps you to efficiently parse the format as it is streamed in real-time. JSON is also a better-suited data format for the unstructured dataset.*

9. A large language model (LLM) for natural language processing (NLP) will be deployed. The model requires fast Network File System (NFS) access to a large dataset from multiple instances.  
Which AWS storage option is best suited for storing the data during training?
  
    *Amazon EFS provides a scalable, elastic NFS file system that can be mounted to multiple Amazon EC2 instances. It is ideal for sharing large datasets across multiple instances that train a machine learning model in parallel.*

10. A data analyst is examining a dataset intended for future use in a machine learning (ML) model and is performing exploratory data analysis. The dataset contains information about customer age, income, and spending data.  
Which type of visualization would help the data analyst determine relationships between customer age and income?
  
    *Scatterplots can visualize relationships between two different numeric variables. With this visualization method, you can view patterns between multiple variables.*

11. A data engineer is working on a machine learning (ML) project that requires real-time data processing for model inference. The data team needs to ingest and process large volumes of streaming data from various sources, such as social media and application clickstream data.  
Which AWS streaming services would be best suited for processing real-time streaming data for the ML inference with minimal management overhead?
  
    *Kinesis Data Streams provides durable real-time data streaming that can capture and store data from many different sources. Amazon Managed Service for Apache Flink can query, analyze, and run computations on streaming data. Using a combination of both of these services, you can ingest real-time data using Kinesis Data Streams. Then, you can process it with Apache Flink for suitable use for ML inference.*