CREATE TABLE IF NOT EXISTS {iceberg_table} (
                    `VendorID` DOUBLE,
                    `tpep_pickup_datetime` TIMESTAMP,
                    `tpep_dropoff_datetime` TIMESTAMP,
                    `passenger_count` DOUBLE,
                    `trip_distance` DOUBLE,
                    `RatecodeID` DOUBLE,
                    `store_and_fwd_flag` STRING,
                    `PULocationID` BIGINT,
                    `DOLocationID` BIGINT,
                    `payment_type` DOUBLE,
                    `fare_amount` DOUBLE,
                    `extra` DOUBLE,
                    `mta_tax` DOUBLE,
                    `tip_amount` DOUBLE,
                    `tolls_amount` DOUBLE,
                    `improvement_surcharge` DOUBLE,
                    `total_amount` DOUBLE,
                    `year` BIGINT,
                    `month` BIGINT
                )
                USING iceberg;
                
 
INSERT OVERWRITE iceberg.hoangnn7.{year}_{month}
(`VendorID`, `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `passenger_count`, `trip_distance`,
 `RatecodeID`, `store_and_fwd_flag`, `PULocationID`, `DOLocationID`, `payment_type`,
 `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`,
 `improvement_surcharge`, `total_amount`, `year`, `month`)
SELECT
    `VendorID`, `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `passenger_count`, `trip_distance`,
    `RatecodeID`, `store_and_fwd_flag`, `PULocationID`, `DOLocationID`, `payment_type`,
    `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`,
    `improvement_surcharge`, `total_amount`, `year`, `month`
FROM default.demo_dbdata_tmp;
