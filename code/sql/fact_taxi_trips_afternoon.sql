CREATE TABLE iceberg.hoangnn7.fact_taxi_trip_evening_{year} (
    LocationID DOUBLE,
    VendorID DOUBLE,
    PaymentID DOUBLE,
    feature STRING, 
    trip_count BIGINT, 
    avg_total_amount DOUBLE, 
    avg_trip_distance DOUBLE,
    avg_tip_amount DOUBLE, 
    avg_passenger_count BIGINT,
    avg_trip_duration_minutes DOUBLE,
    max_trip_distance DOUBLE,
    min_trip_distance DOUBLE, 
    max_trip_duration_minutes DOUBLE, 
    min_trip_duration_minutes DOUBLE,
    amount DOUBLE
) USING iceberg;

INSERT INTO iceberg.hoangnn7.fact_taxi_trip_evening_{year}
SELECT 
    dl.Borough AS LocationID,
    dv.VendorName AS VendorID,
    dpt.PaymentType AS PaymentID,
    CONCAT(dl.Borough,'_', dv.VendorName, '_', dpt.PaymentType) AS feature,
    COUNT(*) AS trip_count,
    AVG(y.total_amount) AS avg_total_amount,
    AVG(y.Trip_distance) AS avg_trip_distance,
    AVG(y.Tip_amount) AS avg_tip_amount,
    ROUND(AVG(y.Passenger_count), 0) AS avg_passenger_count,
    AVG((unix_timestamp(y.tpep_dropoff_datetime) - unix_timestamp(y.tpep_pickup_datetime))/60) AS avg_trip_duration_minutes,
    MAX(y.Trip_distance) AS max_trip_distance,
    MIN(y.Trip_distance) AS min_trip_distance,
    MAX((unix_timestamp(y.tpep_dropoff_datetime) - unix_timestamp(y.tpep_pickup_datetime))/60) AS max_trip_duration_minutes,
    MIN((unix_timestamp(y.tpep_dropoff_datetime) - unix_timestamp(y.tpep_pickup_datetime))/60) AS min_trip_duration_minutes,
    SUM(y.total_amount) AS amount
FROM year_{year} y
JOIN dim_location dl ON y.PULocationID = dl.LocationID
JOIN dim_vendor dv ON y.VendorID = dv.VendorID
JOIN dim_payment_type dpt ON y.Payment_type = dpt.payment_type
WHERE EXTRACT(HOUR FROM y.tpep_pickup_datetime) BETWEEN 12 AND 17
GROUP BY 
    dl.Borough,
    dv.VendorName,
    dpt.PaymentType;
