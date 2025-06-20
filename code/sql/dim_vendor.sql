CREATE TABLE IF NOT EXISTS iceberg.{schema_name}.dim_vendor (
    VendorID DOUBLE,
    VendorName STRING
)
USING iceberg;

INSERT OVERWRITE iceberg.{schema_name}.dim_vendor VALUES
(1.0, 'Creative Mobile Technologies, LLC'),
(2.0, 'VeriFone Inc.');
