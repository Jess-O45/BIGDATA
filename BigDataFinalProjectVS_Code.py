from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Start Spark
spark = SparkSession.builder \
    .appName("DataJobsMLPrediction") \
    .enableHiveSupport() \
    .getOrCreate()

# Load correct columns from Hive
datajobs = spark.sql("""
SELECT lower_salary,
       upper_salary,
       avg_salary_k,
       python,
       hadoop
FROM data_jobs_income
""")

# Drop rows only where target is null
datajobs = datajobs.dropna(subset=["avg_salary_k"])

# Fill NULL feature columns with 0
datajobs = datajobs.fillna({
    "lower_salary": 0,
    "upper_salary": 0,
    "python": 0,
    "hadoop": 0
})


# Rename target
datajobs = datajobs.withColumnRenamed("avg_salary_k", "label")

# Assemble only numeric features
assembler = VectorAssembler(
    inputCols=["lower_salary", "upper_salary", "python", "hadoop"],
    outputCol="features"
)

assembled_df = assembler.transform(datajobs).select("features", "label")

# Train/Test Split
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Train Linear Regression
lr = LinearRegression()
lr_model = lr.fit(train_data)



# Evaluate
test_results = lr_model.evaluate(test_data)

print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# Write metrics to HBase
# Write metrics to HBase
data = [
    ('metrics1', 'salary:rmse', str(test_results.rootMeanSquaredError)),
    ('metrics1', 'salary:r2', str(test_results.r2)),
]


def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')  # hostname from hostname -f
    connection.open()
    table = connection.table('data_jobs')       # your existing table
    for row in partition:
        row_key, column, value = row
        table.put(row_key.encode(), {column.encode(): value.encode()})
    connection.close()
    
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

spark.stop()