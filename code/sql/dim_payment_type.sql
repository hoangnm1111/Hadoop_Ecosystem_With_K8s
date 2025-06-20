CREATE TABLE IF NOT EXISTS iceberg.{schema_name}.dim_payment_type (
    PaymentID DOUBLE,
    PaymentType STRING
)
USING iceberg;

INSERT OVERWRITE iceberg.{schema_name}.dim_payment_type VALUES
    (1.0, 'Credit card'),
    (2.0, 'Cash'),
    (3.0, 'No charge'),
    (4.0, 'Dispute'),
    5.0, 'Unknown'),
    (6.0, 'Voided trip');
