CREATE TABLE IF NOT EXISTS {iceberg_yearly_table} (
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



INSERT INTO {iceberg_yearly_table}
SELECT * FROM {iceberg_table};
