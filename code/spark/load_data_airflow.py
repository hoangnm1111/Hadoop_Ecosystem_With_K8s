from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, explode, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DoubleType, TimestampType
import threading
from sys import argv

if __name__ == "__main__":
    year = int(argv[1])
    month = int(argv[2])

    spark = SparkSession.builder \
        .appName("Load data") \
        .config("spark.streaming.stopGracefullyOnShutdown", True) \
        .getOrCreate()

    if year == 2020 and month >= 1:
        pass

    else:
        df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:9092") \
            .option("subscribe", f"taxi_data_{year}") \
            .load()

        json_df = df.selectExpr("CAST(key AS STRING) as msg_key", "CAST(value AS STRING) as msg_value")

       

        json_schema = StructType([
            StructField('VendorID', DoubleType(), True),
            StructField('tpep_pickup_datetime', TimestampType(), True),
            StructField('tpep_dropoff_datetime', TimestampType(), True),
            StructField('passenger_count', DoubleType(), True),
            StructField('trip_distance', DoubleType(), True),
            StructField('RatecodeID', DoubleType(), True),
            StructField('store_and_fwd_flag', StringType(), True),
            StructField('PULocationID', IntegerType(), True),
            StructField('DOLocationID', IntegerType(), True),
            StructField('payment_type', DoubleType(), True),
            StructField('fare_amount', DoubleType(), True),
            StructField('extra', DoubleType(), True),
            StructField('mta_tax', DoubleType(), True),
            StructField('tip_amount', DoubleType(), True),
            StructField('tolls_amount', DoubleType(), True),
            StructField('improvement_surcharge', DoubleType(), True),
            StructField('total_amount', DoubleType(), True),
            StructField('year', IntegerType(), True),
            StructField('month', IntegerType(), True)
        ])

        array_schema = ArrayType(json_schema)

        #  Parse chuỗi JSON thành mảng các đối tượng
        parsed_df = json_df.withColumn("parsed_array", from_json(col("msg_value"), array_schema))

        # Explode để chuyển mỗi phần tử trong mảng thành một hàng riêng biệt
        exploded_df = parsed_df.withColumn("parsed_element", explode(col("parsed_array")))

        final_df = exploded_df.select(
            col("parsed_element.VendorID").alias("VendorID"),
            col("parsed_element.tpep_pickup_datetime").alias("tpep_pickup_datetime"),
            col("parsed_element.tpep_dropoff_datetime").alias("tpep_dropoff_datetime"),
            col("parsed_element.passenger_count").alias("passenger_count"),
            col("parsed_element.trip_distance").alias("trip_distance"),
            col("parsed_element.RatecodeID").alias("RatecodeID"),
            col("parsed_element.store_and_fwd_flag").alias("store_and_fwd_flag"),
            col("parsed_element.PULocationID").alias("PULocationID"),
            col("parsed_element.DOLocationID").alias("DOLocationID"),
            col("parsed_element.payment_type").alias("payment_type"),
            col("parsed_element.fare_amount").alias("fare_amount"),
            col("parsed_element.extra").alias("extra"),
            col("parsed_element.mta_tax").alias("mta_tax"),
            col("parsed_element.tip_amount").alias("tip_amount"),
            col("parsed_element.tolls_amount").alias("tolls_amount"),
            col("parsed_element.improvement_surcharge").alias("improvement_surcharge"),
            col("parsed_element.total_amount").alias("total_amount"),
            col("parsed_element.year").alias("year"),
            col("parsed_element.month").alias("month")
        )

        writing_df = final_df.writeStream \
            .format("parquet") \
            .option("format", "append") \
            .option("path", "hdfs://hadoop-hadoop-hdfs-nn:9000/raw/" + str(year) + "/" + str(month)) \
            .option("checkpointLocation", "hdfs://hadoop-hadoop-hdfs-nn:9000/tmp/" + str(year) + "/" + str(month)) \
            .outputMode("append") \
            .start()


        def stop_query():
            writing_df.stop()


        # timer = threading.Timer(24 * 60 * 60, stop_query)
        timer = threading.Timer(40 * 60 * 60, stop_query)
        timer.start()

        writing_df.awaitTermination()

