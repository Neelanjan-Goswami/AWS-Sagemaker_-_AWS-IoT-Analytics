
#70 DATA FROM TARINING & 30% data for vaildation 
#Code-1---------------------------------------------------------------------------------------------------------------

from sagemaker import get_execution_role

#Bucket location to save your custom code in tar.gz format.
custom_code_upload_location = 's3://sagemaker-02122018/customcode/tensorflow_iris'

#Bucket location where results of model training are saved.
model_artifacts_location = 's3://sagemaker-02122018/artifacts'

#IAM execution role that gives SageMaker access to resources in your AWS account.
role = get_execution_role()#The get_execution_role function retrieves the IAM role you created at the time of creating your notebook instance. 

#Code-2---------------------------------------------------------------------------------------------------------------


 !cat "iris_dnn_classifier.py"
 
 #Code-3---------------------------------------------------------------------------------------------------------------
 from sagemaker.tensorflow import TensorFlow #Importing tensorflow class from sagemaker

iris_estimator = TensorFlow(entry_point='iris_dnn_classifier.py',  #Creating Estimator,etreing python scrypt
                            role=role, #IAM Role
                            framework_version='1.11.0',
                            output_path=model_artifacts_location,
                            code_location=custom_code_upload_location,
                            train_instance_count=1, #no of instences
                            train_instance_type='ml.c4.xlarge',
                            training_steps=1000,  
                            evaluation_steps=100)


									  
									  
									  
#Code-4---------------------------------------------------------------------------------------------------------------
	%%time
import boto3

region = boto3.Session().region_name
train_data_location = 's3://sagemaker-sample-data-{}/tensorflow/iris'.format(region)

iris_estimator.fit(train_data_location)


#Code-5---------------------------------------------------------------------------------------------------------------

%%time

iris_predictor = iris_estimator.deploy(initial_instance_count=1,
                                       instance_type='ml.m4.xlarge')
									   
									   
									   
									   
#Code-6---------------------------------------------------------------------------------------------------------------	


iris_predictor.predict([7.0, 3.2, 4.7, 1.4]) #expected label to be 1



								   




							  