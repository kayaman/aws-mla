# Bedrock

Learn more about [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html).

---

To enhance a user question, you can add relevant retrieved documents into the context. You can use prompt engineering techniques to help support effective communication with the LLMs. By augmenting the prompt, the LLMs are able to generate precise answers to user queries.  
Learn more about [augmenting the LLM prompt](https://aws.amazon.com/what-is/retrieval-augmented-generation/).

---

RAG is an approach to retrieve relevant passages of text from a database and use the retrieved text to augment the generation of model responses. You can use Amazon Bedrock knowledge bases to build a repository of information for RAG applications. To use a knowledge base in Amazon Bedrock, you must first create an S3 bucket to use as the data source for the knowledge base. Next, you must create and configure the knowledge base. Finally, you can ingest the data into the knowledge base.  
You should not upload the data to OpenSearch Service. The data source needs to be an S3 bucket to work with Amazon Bedrock knowledge bases. You cannot select OpenSearch Service as the data source for the knowledge base.  

Step 1: Create an Amazon S3 bucket.  
Step 2: Create and configure a knowledge base in Amazon Bedrock and select the Amazon S3 bucket as the data source.  
Step 3: Ingest the data into the knowledge base.

Learn more about [Amazon Bedrock knowledge bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html).  
Learn more about [RAG](https://aws.amazon.com/what-is/retrieval-augmented-generation/).

---

Knowledge bases for Amazon Bedrock provide you with the capability to amass data sources into a repository of information. With knowledge bases, you can build an application that uses RAG. You can use provisioned throughput mode to purchase model units for a specific base or custom model. Provisioned throughput mode is most suitable for large, consistent inference workloads that need assured throughput. In this use case, the chatbot will be used consistently and must be available at all times. Using provisioned throughput mode is the most cost-effective solution because it provides throughput for Amazon Bedrock permanently.  
Learn more about [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html).  
Learn more about [provisioned throughput for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/prov-throughput.html).  
Learn more about [knowledge bases for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html).  

---

Step 1: Prepare a dataset of labeled entries in JSON Lines format.  
Step 2: Run a fine-tuning job by using a source model and the training data.  
Step 3: Deploy the fine-tuned model that includes provisioned throughput.  
Amazon Bedrock offers two methods for model customization: fine-tuning and continued pre-training. You can use fine-tuning to specialize a foundation model (FM) on a specific task by using labeled data. Labeled data must be prepared with prompts and completions specific to that task. Continued pre-training uses unlabeled data to familiarize a model with a domain without specializing in any task in particular. Because the company wants to improve performance in one specific task (summarization), you should use fine-tuning. After you train a custom model, you need to purchase provisioned throughput for the custom model before you can start using the model. Serverless inference is not supported for custom models.  
Learn more about [model customization in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html).  
Learn more about [how to use a custom model](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-use.html). 