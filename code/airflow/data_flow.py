from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from pyhive import hive
from datetime import timedelta
from pyspark.sql import SparkSession
from confluent_kafka import Producer
import requests
import json
import os
import pandas as pd  # Import pandas để xử lý file CSV cục bộ
import time
import subprocess

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": "2018-12-31 00:00:00",
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG("data_flow_finaly", default_args=default_args, schedule_interval="0 0 1 * *")

spark = SparkSession.builder.appName("Get time from HDFS").getOrCreate()
df = spark.read.option("header", "true").csv("hdfs://hadoop-hadoop-hdfs-nn:9000/time")
time = df.first()
year = int(time["year"])
month = int(time["month"])
def increase_time_def():
    global year
    global month

    if year == 2020 and month == 7:
        pass
    else:
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        
        columns = ["year", "month"]
        data = [(year, month)]
        df = spark.createDataFrame(data, columns)
        df.write.option("header", "true").mode("overwrite").csv("hdfs://hadoop-hadoop-hdfs-nn:9000/time")


def on_delivery(err, msg):
    if err is not None:
        print(f"Message error: {err}")
    else:
        print(f"Messasge success {msg.topic()} with key: {msg.key().decode('utf-8')}")

def extract_data():
    time.sleep(30)
    global year, month

    kafka_config = {
        'bootstrap.servers': 'kafka:9092',
        'acks': 'all'
    }

    kafka_producer = Producer(kafka_config)
    kafka_topic = f"taxi_data_{year}"
    api_endpoint = "http://service:5000/api/get_taxi_data"

    offset = 0
    batch_size = 100

    while True:
        query_params = {
            'year': year,
            'month': month,
            'offset': offset,
            'limit': batch_size
        }

        try:
            response = requests.get(api_endpoint, params=query_params)
            result = response.json()

            status = result.get('status')
            records = result.get('data')

            if status == 'error':
                print("API error")
                break

            message_key = f"{year}_{month}_{offset}"
            message_value = json.dumps(records)

            kafka_producer.produce(
                topic=kafka_topic,
                key=message_key,
                value=message_value,
                callback=on_delivery
            )

            if status == 'complete':
                print("Complete fetch data")
                break

            offset += batch_size

        except Exception as ex:
            print(f"Exception: {ex}")
            break

    kafka_producer.flush()
    
ef get_spark_thrift_conn(hive_server2_conn_id: str = "hiveserver2_default_1") -> hive.Connection:
    conn_query_params = BaseHook.get_connection(hive_server2_conn_id)

    #extra = json.loads(conn_query_params.get_extra() or "{}")
    #password = conn_query_params.password if conn_query_params.login else None
    conn = hive.connect(
        host=conn_query_params.host,
        database=conn_query_params.schema,
        port=conn_query_params.port,
        username=conn_query_params.login,
        password=None,
        auth="NONE"
    )
    print("NOTICE: Please close conn after using or use with () statement for auto-closing!")
    return conn

def query_iceberg():
    global year
    global month

    if year == 2020 and month == 7:
        return
    else:
        table_suffix = f"{year}_{month}" 
        schema_name = "hoangnn7"

        sql_scripts_dir = "sql/"

        sql_files = [
            "create_tmp_table.sql",
            "taxi_data_table.sql",
            "yearly_table.sql",
            "dim_vendor.sql",
            "dim_payment_type.sql",
            "dim_location.sql",
            "top_location.sql",
            "fact_taxi_trips_morning.sql",
            "fact_taxi_trips_afternoon.sql",
            "fact_taxi_trips_evening.sql"
            "fact_taxi_trips_weekend_morning.sql",
            "fact_taxi_trips_weekend_afternoon.sql",
            "fact_taxi_trips_weekend_evening.sql"
        ]

        with get_spark_thrift_conn() as conn:
            cursor = conn.cursor()
            for sql_file in sql_files:
                file_path = os.path.join(sql_scripts_dir, sql_file)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        sql_content = f.read()
                        sql_content = sql_content.format(year=year, month=month)
                        cursor.execute(sql_content)
                else:
                    raise FileNotFoundError(f"SQL file {file_path} not found")
                    
extract_data = PythonOperator(
    task_id="extract_data",
    python_callable=extract_data,
    dag=dag
)

load_data = BashOperator(
    task_id="load_data",
    bash_command=f"spark-submit /home/spark/load_data_airflow.py {year} {month}",
    dag=dag
)

transform_data = BashOperator(
    task_id="transform_data",
    bash_command=f"spark-submit /home/spark/transform_data_airflow.py {year} {month}",
    dag=dag
)

query_task = PythonOperator(
    task_id='query_iceberg_demo',
    python_callable=query_iceberg,
    dag=dag,
)

increase_time = PythonOperator(
    task_id="increase_time",
    python_callable=increase_time_def,
    dag=dag
)

[extract_data, load_data] >> transform_data >> query_task >> increase_time
