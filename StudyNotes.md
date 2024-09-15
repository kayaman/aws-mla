# Study Notes


## IAM


IAM roles provide permissions to AWS services or users to access AWS resources securely. These roles are used to delegate access within an AWS account or across different AWS accounts. When assuming an IAM role, a user or service temporarily takes on the permissions and policies that are associated with that role. This action gives the user or service the ability to perform actions on AWS resources that are based on the permissions that are granted by the role without the need to use long-term credentials, such as access keys.

To provide access to a user in one AWS account (the ML startup's account) to resources in another AWS account (the company's account), you must create an IAM role in the company's account with the necessary permissions and trust relationship and then specify the ML startup account's ID. The user in the client account can then assume the role and obtain temporary credentials for secure cross-account access. Configuring cross-account IAM roles is the only way to provide both programmatic and console access to S3 buckets across accounts. In this scenario, the role that is created in the company's account is then assumed by the ML startup's users to access the S3 bucket.

## S3

You can use Amazon S3 Event Notifications to receive notifications when predefined events occur in an S3 bucket. You can use event notifications to invoke an event. In this scenario, you can use the event to run a step function as the destination.

- https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventNotifications.html

## Step Functions

Step Functions is a serverless orchestration service that you can use to coordinate and sequence multiple AWS services into serverless workflows.

- https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html

## Glue

AWS Glue is a serverless integration service you can use to integrate data from multiple sources, including Amazon S3. AWS Glue can be used to interact with the Apache Spark framework engine for executing data engineering and ML processes. You can create an AWS Glue job to automate the extract, transform, and load (ETL) processes. This answer meets the requirements with the least operational effort.  
Learn more about [the PySpark extension for Python in AWS Glue](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python-extensions.html).  
Learn more about [how to work with Apache Spark jobs in AWS Glue](https://docs.aws.amazon.com/glue/latest/dg/etl-jobs-section.html).


## Databases

A vector database is a type of database that can handle vector data. Vector data often consists of embeddings that are used in ML models. Vector databases provide fast similarity search and retrieval based on vector proximity. A relational database is a database that is structured to recognize relationships among stored items of information. Relational databases typically use tables with rows and columns. You can use relational databases to handle structured data that has defined relationships.  

You should store customer information in a relational database because one customer is associated with attributes such as name, address, or phone number. You should store movie and actor information in a relational database. In movie and actor data, each entity is associated with attributes. There is a many-to-many relationship between movies and actors that you can model in a mapping table.  

You should store text embeddings for RAG in a vector database. You can query the vector database for relevant text in use cases such as question-answering. You should store product information for generative AI–powered recommendations in a vector database. In a vector database, applications can store information about products in vectors. Then, the vector database can make recommendations by searching for products that are similar to a product by using the vectors.  
Learn more about [vector databases on AWS](https://aws.amazon.com/what-is/vector-databases/).  
Learn more about [relational databases on AWS](https://aws.amazon.com/what-is/vector-databases/).  
Learn more about [RAG](https://aws.amazon.com/what-is/retrieval-augmented-generation/).  

## Evaluation Metrics

Precision is a valid metric for binary classification problems. Precision can help quantify the proportion of positive predictions that were identified correctly. You can use precision when you want to minimize false positives. In this scenario, you want to ensure that legitimate emails are not incorrectly flagged as spam (a false positive result). Therefore, precision would be the most suitable model evaluation metric to use.  
Learn more about [metrics and validation in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-metrics-validation.html).

## Correlation Metrics

- The chi square is a statistical test assessing association between categorical variables. Chi square is not used to assess correlation, and is not suitable to assess the relationships between numeric features.
- The Spearman coefficient assesses correlation between numeric features. The Spearman coefficient does not assume linear relationship. Therefore, the Spearman coefficient can be used to assess the non-linear relationship between numeric features.
- The phi coefficient assesses correlation between binary variables. The phi coefficient is not suitable to assess the relationships between numeric features.
- The Pearson coefficient assesses linear relationship between numeric features. Because the Pearson coefficient assumes linear relationship, the metric can be biased when the relationship between variables is non-linear. Therefore, the Pearson coefficient is not suitable to assess the non-linear relationships between numeric features.  
  
Learn more about [correlation metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas-explore-data-analytics.html).

## Computing

Accelerated computing group of ML instances. Accelerated computing EC2 instance types are equipped with accelerators such as GPUs or inferential chips. You can use this solution to accelerate inference of deep learning models. Because the use case of text summarization and text generation requires large deep learning models, this group of instances is the most suitable solution for inference.  
Learn more about [accelerated computing instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html).  
Learn more about [how to choose instance types for large model inference](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-choosing-instance-types.html).


## Training

Increase the amount of domain-specific features in the training dataset. When you are training a new ML model, if you identify that the model is not accurately capturing the underlying relationship in the training dataset (in this scenario, to identify a product that is defective or not), it means that the model is underfitting. Because the model is not accurately capturing the relationship between the input examples and target values, the model will not perform well when tested on the evaluation data. To solve the underfitting problem, you should increase the model fit to better capture the relationship between the inputs examples and targets. You can increase the model fit by increasing the amount of domain-specific features that are in the training dataset.  
Learn more about [model fit underfitting and overfitting](https://aws.amazon.com/what-is/overfitting/).

---

Regularization penalizes extreme weight values in model coefficients.  
L1 regularization is used to reduce the noise in the models, reducing some model coefficient weights to 0. When regularization is used, it effectively eliminates the features and overall complexity of the model, and it can help to solve overfitting models. However, this model is underfitting, so increasing L1 regularization will not help in this scenario.  
L2 regularization reduces the weights of the model coefficients, but does not push the weights to 0. Regularization is used to stabilize the weights when there is a high correlation between the input features, and it can help to solve overfitting models. However, this model is underfitting, so increasing L2 regularization does not help in this scenario.  
Learn more about [regularization type and amount](https://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters.html#regularization-type-and-amount).
 
---

An error rate that rapidly decreases as the epochs increases indicates an overfitting problem when training the model. This problem means that the model is memorizing rather than generalizing, and the model is too dependent on the training data. Overfitting causes the model to generalize poorly when presented to new data not present in the initial training dataset. Regularization helps reduce the overfitting of the model because regularization penalizes extreme weight values in model parameters. This method helps the model to generalize better and avoid the initial overfitting issue encountered.  
Learn more about [overfitting](https://aws.amazon.com/what-is/overfitting/).

---

Re-train the model periodically as the products that are offered change and the inventory changes. Use the original training data and new training data. As inventory and products change over time, the distribution of data that is used to make predictions will drift. The most suitable solution would be to re-train the ML model on a regular basis by using the original training data and newly collected data.  
Learn more about [how to automate model re-training by using SageMaker workflows](https://docs.aws.amazon.com/sagemaker/latest/dg/workflows.html).

## Costs


Assign a user-defined tag to the project AWS resources that includes a project identifier. Activate user-defined tags in the AWS Billing and Cost Management console and use AWS Cost Explorer to filter costs by the project identifier.
Cost Explorer provides a visual interface to track the costs of AWS resource usage. AWS resources that you use in this solution support the addition of user-defined tags. If you add a project identifier tag to these resources and activate user-defined tags in the Billing and Cost Management console, you can use the project identifier filter in Cost Explorer to track the project costs.  
Learn more about [how to tag AWS resources](https://docs.aws.amazon.com/tag-editor/latest/userguide/tagging.html).  
Learn more about [how to activate and filter costs by using user-defined tags](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/cost-alloc-tags.html).  

## Responsible AI

- [Responsible AI](https://aws.amazon.com/ai/responsible-ai/)
- [Tools and resources to build AI responsibly](https://aws.amazon.com/ai/responsible-ai/resources/)