CREATE TABLE IF NOT EXISTS iceberg.{schema_name}.dim_location (
    LocationID DOUBLE,
    Borough STRING,
    Zone STRING,
    service_zone STRING
)
USING iceberg;

CREATE TABLE IF NOT EXISTS default.location_dbdata_tmp
USING org.apache.spark.sql.parquet
LOCATION 'hdfs:///location/*.snappy.parquet';

INSERT OVERWRITE iceberg.{schema_name}.dim_location (LocationID, Borough, Zone, service_zone) 
SELECT `_c0`, `_c1`, `_c2`, `_c3`
FROM default.location_dbdata_tmp;
