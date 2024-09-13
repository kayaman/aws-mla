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

## SageMaker

### VPC

- https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html

- Option: Configure SageMaker in VPC only mode. Configure security groups to block internet access.
  You can use a VPC to launch AWS resources within your own isolated virtual network. Security groups are a security control that you can use to control access to your AWS resources. You can protect your data and resources by managing security groups and restricting internet access from your VPC. However, this solution requires additional network configuration and therefore increases operational overhead.  
  Learn more about [SageMaker in VPC only mode](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-vpc.html).

### Deployment

You can use a SageMaker asynchronous endpoint to host an ML model. With a SageMaker asynchronous endpoint, you can receive responses for each request in near real time for up to 60 minutes of processing time. There is no idle cost to operate an asynchronous endpoint. Therefore, this solution is the most cost-effective. Additionally, you can configure a SageMaker asynchronous inference endpoint with a connection to your VPC.

- https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html#deploy-model-options
- https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html

During an in-place deployment, you update the application by using existing compute resources. You stop the current version of the application. Then, you install and start the new version of the application. In-place deployment does not meet the requirement to minimize the risk of downtime because this strategy relies on downtime to make the shift. Additionally, this strategy does not meet the requirement to gradually shift traffic from the old model to the new model.  
Learn more about [in-place deployment](https://docs.aws.amazon.com/whitepapers/latest/introduction-devops-aws/in-place-deployments.html).

SageMaker is a fully managed service for the end-to-end process of building, serving, and monitoring ML models. You can create a SageMaker model resource from an existing model that you built on your own. Then, you can deploy that model to a SageMaker endpoint. Serverless SageMaker endpoints are the most suitable for this scenario and provide the least effort. Serverless SageMaker endpoints scale independently in a fully serverless manner. Additionally, the memory requirements fit within the 6 GB memory and 200 maximum concurrency limits of serverless endpoints.  
Learn more about [SageMaker endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html).  
Learn more about [SageMaker endpoint types](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-deployment.html).  
Learn more about [serverless inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html).

### Containers

You can use the SageMaker SDK to bring existing ML models that are written in R into SageMaker by using the "bring your own container" option. This solution requires the least operational overhead because you only need to compose a Dockerfile for each existing model.  
Learn more about [how to bring your own containers in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-containers.html).  
Learn more about [how to use R in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/r-guide.html).  

### Endpoints

SageMaker endpoints support a one-time or recurring scheduled scaling action to change the minimum and maximum capacity of the SageMaker endpoint. SageMaker also supports target tracking scaling policies to dynamically increase or decrease capacity based on a target value for a performance metric. You can schedule a capacity increase to provision additional endpoint resources before each promotional event, while a target tracking scaling policy is still in effect. This combination of scaling policies provides a consistent experience to the many users that join as the events begin. The target tracking scaling policy will continue to dynamically scale capacity during the event relative to the new minimum and maximum capacity levels.  
Learn more about [how to use scheduled scaling policies for SageMaker endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-prerequisites.html#scheduled-scaling).  
Learn more about [how to use target tracking scaling policies for SageMaker endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-prerequisites.html#endpoint-auto-scaling-policy).

### Input Modes

Input modes include file mode, pipe mode, and fast file mode. File mode downloads training data to a local directory in a Docker container. Pipe mode streams data directly to the training algorithm. Therefore, pipe mode can lead to better performance. Fast file mode provides the benefits of both file mode and pipe mode. For example, fast file mode gives SageMaker the flexibility to access entire files in the same way as file mode. Additionally, fast file mode provides the better performance of pipe mode.

Before you begin training, fast file mode identifies S3 data source files. However, fast file mode does not download the files. Instead, fast file mode gives the model the ability to begin training before the entire dataset has finished loading. Therefore, fast file mode decreases the startup time. As the training progresses, the entire dataset will load. Therefore, you must have enough space within the storage capacity of the training instance. This solution provides an update to only a single parameter and does not require any code changes. Therefore, this solution requires the least operational overhead.

Learn more about [how to access training data](https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html).


### AMT

SageMaker AMT searches for the most suitable version of a model by running training jobs based on the algorithm and objective criteria. You can use a warm start tuning job to use the results from previous training jobs as a starting point. You can set the early stopping parameter to Auto to enable early stopping. SageMaker can use early stopping to compare the current objective metric (accuracy) against the median of the running average of the objective metric. Then, early stopping can determine whether or not to stop the current training job. The TRANSFER_LEARNING setting can use different input data, hyperparameter ranges, and other hyperparameter tuning job parameters than the parent tuning jobs.

### Model Registry

You can use SageMaker Model Registry to create a catalog of models for production, to manage the versions of a model, and to associate metadata to the model. Additionally, SageMaker Model Registry can manage approvals and automate model deployment for continuous integration and continuous delivery (CI/CD). You would not use SageMaker Model Registry for model re-training.

- https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html

### Experiments

SageMaker Experiments is a feature of SageMaker Studio that you can use to automatically create ML experiments by using different combinations of data, algorithms, and parameters. You would not use SageMaker Experiments to collect new data for model re-training.

- https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html

### Model Monitor

You can use SageMaker Model Monitor to effectively gauge model quality. Data Capture is a feature of SageMaker endpoints. You can use Data Capture to record data that you can then use for training, debugging, and monitoring. Then, you could use the new data that is captured by Data Capture to re-train the model. Data Capture runs asynchronously without impacting production traffic.

- https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-faqs.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture.html

You can use the ModelExplainabilityMonitor class to generate a feature attribution baseline and to deploy a monitoring mechanism that evaluates whether the feature attribution has occurred. You can use CloudWatch to send notifications when feature attribution drift has occurred.

Learn more about [how to monitor for feature attribution drift](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html).

Learn more about [the ModelExplainabilityMonitor class](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-shap-baseline.html).

Learn more about [CloudWatch integration with SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html).


### Pipelines


SageMaker Pipelines is a workflow orchestration service within SageMaker. SageMaker Pipelines supports the use of batch transforms to run inference of entire datasets. Batch transforms are the most cost-effective inference method for models that are called only on a periodic basis. Real-time inference would create instances that the company would not use for most of the week.

After you create the inference pipeline, EventBridge can automate the execution of the pipeline. You would need to create a role to allow EventBridge to start the execution of the pipeline that was created in the previous step. You can use a scheduled run to execute the inference pipeline at the beginning of every week. You do not have a specific pattern that you need to match to invoke the execution. Therefore, you do not need to create a custom event pattern.

- https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-batch.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/pipeline-eventbridge.html


### Jobs


You can use SageMaker processing jobs for data processing, analysis, and ML model training. You can use SageMaker processing jobs to perform transformations on images by using a script in multiple programming languages. In this scenario, you can run the custom code on data that is uploaded to Amazon S3. SageMaker processing jobs provide ready-to-use Docker images for popular ML frameworks and tools. Additionally, SageMaker offers built-in support for various frameworks including TensorFlow, PyTorch, scikit-learn, XGBoost, and more.

- https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html


### TensorBoard


SageMaker with TensorBoard is a capability of SageMaker that you can use to visualize and analyze intermediate tensors during model training. SageMaker with TensorBoard provides full visibility into the model training process, including debugging and model optimization. This solution gives you the ability to debug issues, including lower than expected precision for a specific class. You can analyze the intermediate activations and gradients during training. Then, you can gain insights into why some mobile phone images were getting misclassified. Finally, you can make adjustments to improve model performance.

- https://docs.aws.amazon.com/sagemaker/latest/dg/tensorboard-on-sagemaker.html

## Bedrock

- To enhance a user question, you can add relevant retrieved documents into the context. You can use prompt engineering techniques to help support effective communication with the LLMs. By augmenting the prompt, the LLMs are able to generate precise answers to user queries.

Learn more about [augmenting the LLM prompt](https://aws.amazon.com/what-is/retrieval-augmented-generation/).


- You can use PDPs and Shapley values for model interpretability in ML. Shapley values focus on feature attribution. PDPs illustrate how the predicted target response changes as a function of one particular input feature of interest. DPL is a metric that you can use to detect pre-training bias. You can use DPL to avoid ML models that could potentially be biased or discriminatory.

Learn more about [Shapley values](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-shapley-values.html).

Learn more about [PDPs](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-processing-job-analysis-results.html#clarify-processing-job-analysis-results-pdp).

Learn more about [DPL](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-true-label-imbalance.html).


## Amazon Comprehend


Amazon Comprehend can be used to detect and redact personal information from user interactions. Amazon Comprehend provides the ability to locate and redact PII entities in English or Spanish text documents. By leveraging Amazon Comprehend, you can easily process and anonymize personal information in the customer data platform.

- https://docs.aws.amazon.com/comprehend/latest/dg/how-pii.html


## Responsible AI

- [Responsible AI](https://aws.amazon.com/ai/responsible-ai/)
- [Tools and resources to build AI responsibly](https://aws.amazon.com/ai/responsible-ai/resources/)