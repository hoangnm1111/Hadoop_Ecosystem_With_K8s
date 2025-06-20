from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from sys import argv

if __name__ == "__main__":
    year = int(argv[1])
    month = int(argv[2])

    spark = SparkSession.builder.appName("Transform data").getOrCreate()

    if year == 2020 and month >= 1:
        pass

    else:
        df = spark.read.parquet("hdfs://hadoop-hadoop-hdfs-nn:9000/raw/" + str(year) + "/" + str(month))

        # Process duplicated rows
        df_duplicate = df.dropDuplicates()

        # Lấy danh sách các cột, loại trừ 'tpep_pickup_datetime' và 'tpep_dropoff_datetime'
        columns_to_check = [c for c in df_duplicate.columns if
                            c not in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']]

        # Kiểm tra số dòng có missing value, chỉ trên các cột được chọn
        missing_count = df_duplicate.select(
            [count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in columns_to_check]
        ).collect()[0].asDict()

        # Xóa các dòng có missing value, chỉ xét các cột không bị loại trừ
        df_clean = df_duplicate.dropna(subset=columns_to_check)

        # Them cot xu li ngay gio
        df_clean1 = df_clean.withColumn("pickup_date", to_date(col("tpep_pickup_datetime")))
        df_clean1 = df_clean1.withColumn("pickup_time", date_format(col("tpep_pickup_datetime"), "HH:mm:ss"))

        df_clean1 = df_clean1.withColumn(
            "trip_duration",
            (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60
        )

        # Tính tổng các khoản vào cột tạm thời
        df_clean1 = df_clean1.withColumn(
            "calculated_total",
            col("fare_amount") + col("extra") + col("mta_tax") +
            col("tip_amount") + col("tolls_amount") + col("improvement_surcharge")
        )

        # Áp dụng các quy tắc lọc dữ liệu
        df_filtered = df_clean1.filter(
            # Điều kiện 1: trip_distance > 0 và <= 100
            (col("trip_distance") >= 0) & (col("trip_distance") <= 100) &

            # Điều kiện 2: Thời gian di chuyển hợp lệ
            (col("tpep_dropoff_datetime") >= col("tpep_pickup_datetime")) &
            (col("trip_duration") < 1440) &
            (col("trip_duration") > 0) &

            # Điều kiện 3: passenger_count > 0 và <= 6
            (col("passenger_count") >= 0) & (col("passenger_count") <= 6) &

            # Điều kiện 4: Các cột tiền không âm
            (col("total_amount") >= 0) &
            (col("fare_amount") >= 0) &
            (col("tip_amount") >= 0) &
            (col("tolls_amount") >= 0) &
            (col("extra") >= 0) &
            (col("mta_tax") >= 0) &
            (col("improvement_surcharge") >= 0) &

            # Điều kiện 5: Tổng các khoản bằng total_amount
            (
                    col("calculated_total") == col("total_amount")
            ) &

            ~((col("passenger_count") == 0) & (col("total_amount") != 0))

        )

        # Xóa cột 'tpep_pickup_datetime' và 'trip_duration' (không cần nữa)
        df_filtered = df_filtered.drop("pickup_date", "pickup_time", "trip_duration", "calculated_total")


        # Define the output path
        output_path = f"hdfs://hadoop-hadoop-hdfs-nn:9000/staging/{year}/{month}"

        # Write the cleaned DataFrame to HDFS as parquet
        df_filtered.write.mode("overwrite").parquet(output_path)


