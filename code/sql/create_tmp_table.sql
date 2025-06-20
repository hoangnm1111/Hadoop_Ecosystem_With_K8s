DROP TABLE IF EXISTS default.demo_dbdata_tmp;

CREATE DATABASE IF NOT EXISTS iceberg.{schema_name};

CREATE TABLE IF NOT EXISTS default.demo_dbdata_tmp
USING org.apache.spark.sql.parquet
LOCATION '{staging_path}
