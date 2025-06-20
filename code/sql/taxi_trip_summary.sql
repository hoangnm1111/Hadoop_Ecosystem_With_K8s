CREATE TABLE IF NOT EXISTS iceberg.hoangnn7.taxi_trip_summary (
    Ngay INT,
    Thang INT,
    Nam INT,
    Gio INT,
    Quan STRING,
    So_chuyen BIGINT,
    ThoiGian STRING
)
USING iceberg;

INSERT INTO iceberg.hoangnn7.taxi_trip_summary
SELECT 
    DAY(tpep_pickup_datetime) AS Ngay,
    MONTH(tpep_pickup_datetime) AS Thang,
    YEAR(tpep_pickup_datetime) AS Nam,
    HOUR(tpep_pickup_datetime) AS Gio,
    dl.Borough AS Quan,
    COUNT(*) AS So_chuyen,
    CONCAT(
        CAST(YEAR(tpep_pickup_datetime) AS STRING), '-',
        LPAD(CAST(MONTH(tpep_pickup_datetime) AS STRING), 2, '0'), '-',
        LPAD(CAST(DAY(tpep_pickup_datetime) AS STRING), 2, '0')
    ) AS ThoiGian
FROM iceberg.hoangnn7.all_taxi_data atd
LEFT JOIN dim_location dl
    ON atd.PULocationID = dl.LocationID
GROUP BY 
    DAY(tpep_pickup_datetime),
    MONTH(tpep_pickup_datetime),
    YEAR(tpep_pickup_datetime),
    HOUR(tpep_pickup_datetime),
    dl.Borough;
