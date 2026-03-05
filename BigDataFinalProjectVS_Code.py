from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Start Spark
spark = SparkSession.builder.appName("DataJobsMLPrediction").enableHiveSupport().getOrCreate()

sc = spark.sparkContext

#load the data from Hive table 'data_jobs_income' into spark 
datajobs = spark.sql("SELECT job_title, location, lower_salary, upper_salary, avg_salary_k, python, hadoop FROM data_jobs_income")

#drop NAs
datajobs = datajobs.na.drop()

#prepare the data for MLlib by assembling features into a vector
assembler = VectorAssembler(
    inputCols=["lower_salary", "upper_salary", "python", "hadoop"],
    outputCol="features"
)
                        
assembled_df = assembler.transform(datajobs).select("features", "label")

#Split the data into training and testing sets
train_data, test_data = assembled_df.randomSplit([0.7,0.3])

#Initialize and train a Linear Regression model
lr = LinearRegression(labelCol="avg_salary_k")
lr_model = lr.fit(train_data)

#Evaluate the model on the test data
test_results = lr_model.evaluate(test_data)

#Print the model performance metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# Write metrics to HBase with happybase populated with the metrics
data = [
    ('metrics1', 'cf:rmse',str(test_results.rootMeanSquaredError)),
    ('metrics1', 'cf:r2', str(test_results.r2)),
]
# Function to write data to HBase inside each partition
def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('my_table') # Update table name 
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

# Parallelize data and apply the function with for each partition
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Stop the Spark session
spark.stop()