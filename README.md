# wine-quality-prediction-ML-AWS

•	Create EMR Cluster which contains 5 nodes (1 Master and 4 Slaves) on AWS.

•	Upload below deliverables to S3 bucket using upload option present on UI.
1.	TrainingDataset.csv
2.	training.py
3.	requirements.txt
4.	ValidationDataset.csv
5.	test.py

•	Login to master node using putty.
•	Download files from bucket to master node using below command:

1.	aws s3 cp s3://aws-logs-658611668728-us-east-1/elasticmapreduce/j-NQXMR7HXYD84/TrainingDataset.csv ./
2.	aws s3 cp s3://aws-logs-658611668728-us-east-1/elasticmapreduce/j-NQXMR7HXYD84/training.py ./
3.	aws s3 cp s3://aws-logs-658611668728-us-east-1/elasticmapreduce/j-NQXMR7HXYD84/requirements.txt ./ 

•	Install all python dependencies which are mentioned in requirements.txt file using below command:

pip install -r requirements.txt

•	To available datasets files to slaves execute below command:

1.	hadoop fs -put TrainingDataset.csv

2.	hadoop fs -put ValidationDataset.csv

•	Run the training model using below command:

spark-submit training.py

•	Create new EC2 instances using AWS management console, login to EC2 instance and perform below steps on EC2 instance.

•	Install Java and all python dependencies on EC2 instance. 

•	Upload below files to EC2 instance using WinSCP

1.	ValidationDataset.csv
2.	test.py
3.	requirements.txt

•	Upload created training model from S3 bucket to EC2 instance using WinSCP.

•	Create one directory name ‘wine_quality’ on EC2 instance and move data and metadata folders which are downloaded from S3 bucket into it using below command:

1.	mkdir wine_quality
2.	mv data wine_quality
3.	mv metadata wine_quality

•	Install Scala on EC2 instance using below commands:

1.	wget https://downloads.lightbend.com/scala/2.12.4/scala-2.12.4.rpm
2.	sudo yum install scala-2.12.4.rpm

•	Install Spark on EC2 instance using below commands:

1.	wget https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
2.	sudo tar xvf spark-3.3.1-bin-hadoop3.tgz -C /opt
3.	sudo chown -R ec2-user:ec2-user /opt/spark-3.3.1-bin-hadoop3
4.	sudo ln -fs spark-3.3.1-bin-hadoop3 /opt/spark

•	Set below path variables in bash_profile file.
1.	export SPARK_HOME=/opt/spark
2.	PATH=$PATH:$SPARK_HOME/bin
3.	export PATH

•	Execute below command after above step.

  source  ~/.bash_profile

•	Test the accuracy of the dataset on EC2 using below command:

  spark-submit test.py
  
 Execution of project using Docker :
 
 Docker Implementation: - pull the image from my docker repo using this command: docker pull ruchagoundadkarchavan/wine_quality_prediction

Run the docker image. 
