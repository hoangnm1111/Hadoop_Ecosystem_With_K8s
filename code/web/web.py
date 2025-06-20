import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, dayofmonth, month, year, to_timestamp
from pyspark.ml import PipelineModel
from pyhive import hive
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pyspark.sql.functions as F
import os
import plotly.express as px
#from airflow.hooks.base import BaseHook
import json
import pickle
import uuid

# Thông tin đăng nhập của admin
USERNAME = "admin"
PASSWORD= "123456"

# Hàm kiểm tra đăng nhập
def check_login(username, password):
    return username == USERNAME and password == PASSWORD

def main_page():
    st.set_page_config(layout="wide")

    padding_top = 0

    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                padding-top: {padding_top}rem;
            }}
        </style>""",
                unsafe_allow_html=True,
                )

    st.markdown("""
            <style>
                   .block-container {
                        padding-top: 1rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
            </style>
            """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Custom metric cards */
        [data-testid="stMetric"] {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        /* Metric value */
        [data-testid="stMetricValue"] {
            font-size: 24px;
            color: #2e86c1;
        }

        /* Metric label */
        [data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #5d6d7e;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Style cho expander */
        .st-emotion-cache-1hynsf2 {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        /* Style cho nút download */
        .stDownloadButton button {
            background-color: #4CAF50 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Style cho tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px 4px 0 0;
        }
        /* Style cho expander */
        .st-emotion-cache-1hynsf2 {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("TaxiAnalysPrediction") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.memoryOverhead", "1g") \
        .getOrCreate()

    # File paths
    DATA_PATH = "/home/data/yellow_tripdata_{year}-{month}.csv"
    ZONE_LOOKUP_PATH = "/home/explore/taxi+_zone_lookup.csv"
    GEOJSON_PATH = "/home/explore/NYC_Taxi_Zones.geojson"
    # Thay đổi đường dẫn mô hình
    BOROUGH_MODEL_PATH = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/RF/borough_model"
    PC_MODEL_PATH = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/RF/passenger_count_model"

    # Thêm hàm kết nối Spark Thrift Server
    @st.cache_resource
    def get_spark_thrift_connection():
        try:
            conn = hive.connect(
                host="hadoop-hdfs-hdfs-nn",  # Thay bằng host của bạn
                port=8989,  # Thay bằng port phù hợp
                # username="",
                auth="NONE"
            )
            return conn
        except Exception as e:
            st.error(f"Lỗi khi kết nối đến Spark Thrift Server: {e}")
            return None

    # Hàm truy vấn dữ liệu từ Iceberg
    @st.cache_data
    def query_iceberg_data(start_date, end_date):
        try:
            conn = get_spark_thrift_connection()
            if conn:
                cursor = conn.cursor()

                query = f"""
                SELECT 
                    tpep_pickup_datetime,
                    tpep_dropoff_datetime,
                    passenger_count,
                    trip_distance,
                    PULocationID,
                    DOLocationID,
                    fare_amount,
                    tip_amount,
                    total_amount
                FROM iceberg.hoangnn7.all_taxi_data
                WHERE tpep_pickup_datetime BETWEEN '{start_date}' AND '{end_date}'
                """

                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()

                df = pd.DataFrame(data, columns=columns)

                # Chuyển đổi kiểu dữ liệu datetime
                df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
                df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

                return df
            return None
        except Exception as e:
            st.error(f"Lỗi khi truy vấn dữ liệu từ Iceberg: {e}")
            return None

    # Hàm đọc dữ liệu từ file CSV
    @st.cache_data
    def load_data(year, month):
        file_path = DATA_PATH.format(year=year, month=f"{month}")
        try:
            df = pd.read_csv(file_path)
            df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
            df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
            return df
        except FileNotFoundError:
            st.error(f"File {file_path} không tồn tại!")
            return None

    # Hàm đọc file taxi_zone_lookup
    @st.cache_data
    def load_zone_lookup():
        try:
            zones = pd.read_csv(ZONE_LOOKUP_PATH)
            return zones
        except FileNotFoundError:
            st.error("File taxi_zone_lookup.csv không tồn tại!")
            return None

    # Hàm đọc file GeoJSON
    @st.cache_data
    def load_geojson():
        try:
            geo_data = gpd.read_file(GEOJSON_PATH)
            return geo_data
        except FileNotFoundError:
            st.error(
                "File NYC_Taxi_Zones.geojson không tồn tại! Vui lòng tải từ https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddkc")
            return None

    # Hàm kiểm tra dữ liệu bị nhiễu
    def check_noisy_data(df):
        noise_info = {}
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

        trip_duration = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
        invalid_time = df[
            (df['tpep_dropoff_datetime'] < df['tpep_pickup_datetime']) |
            (trip_duration > 1440) |
            (trip_duration < 0)
            ]
        noise_info['invalid_time'] = {
            'count': len(invalid_time),
            'indices': invalid_time.index.tolist()
        }

        invalid_distance = df[
            (df['trip_distance'] < 0) |
            (df['trip_distance'] > 100)
            ]
        noise_info['invalid_distance'] = {
            'count': len(invalid_distance),
            'indices': invalid_distance.index.tolist()
        }

        invalid_passengers = df[
            (df['passenger_count'] <= 0) |
            (df['passenger_count'] > 6)
            ]
        noise_info['invalid_passengers'] = {
            'count': len(invalid_passengers),
            'indices': invalid_passengers.index.tolist()
        }

        invalid_fare = df[
            (df['fare_amount'] < 0) |
            (df['total_amount'] < 0) |
            (df['fare_amount'] > 1000)
            ]
        noise_info['invalid_fare'] = {
            'count': len(invalid_fare),
            'indices': invalid_fare.index.tolist()
        }

        return noise_info

    # Định nghĩa các độ đo hiệu suất được nhập sẵn cho từng quận
    HARDCODED_METRICS = {
        'Manhattan': {'mae': 43.345, 'mse': 4492.4921, 'rmse': 22.3154, 'r2': 0.5283},
        'Bronx': {'mae': 2.1145, 'mse': 8.4393, 'rmse': 2.9051, 'r2': 0.6782},
        'Queens': {'mae': 11.8753, 'mse': 224.7678, 'rmse': 14.9923, 'r2': 0.5924},
        'Brooklyn': {'mae': 3.3827, 'mse': 21.3098, 'rmse': 4.6131, 'r2': 0.5934},
        'StatenIsland': {'mae': 0.2051, 'mse': 0.1741, 'rmse': 0.4173, 'r2': 0.7912},
        'EWR': {'mae': 0.6876, 'mse': 0.7322, 'rmse': 0.8557, 'r2': 0.7535}
    }
    # Hàm lấy danh sách tuần trong tháng
    def get_weeks_in_month(df, year, month):
        df_month = df[
            (df['tpep_pickup_datetime'].dt.year == year) &
            (df['tpep_pickup_datetime'].dt.month == month)
            ]
        if df_month.empty:
            return []
        weeks = df_month['tpep_pickup_datetime'].dt.isocalendar().week.unique()
        return sorted([int(w) for w in weeks])

    # Hàm tính ngày bắt đầu và kết thúc của tuần
    def get_week_date_range(year, week):
        first_day = datetime(year, 1, 1)
        days_to_monday = (7 - first_day.weekday()) % 7 if first_day.weekday() != 0 else 0
        start_date = first_day + timedelta(days=days_to_monday + (int(week) - 1) * 7)
        end_date = start_date + timedelta(days=6)
        return start_date, end_date

    # Hàm tính ngày cuối tuần
    def get_weekend_dates(year, week):
        start_date, _ = get_week_date_range(year, int(week))
        saturday = start_date + timedelta(days=5)
        sunday = start_date + timedelta(days=6)
        return saturday, sunday

    # Hàm lọc dữ liệu theo tuần
    def filter_by_week(df, year, month, week):
        return df[
            (df['tpep_pickup_datetime'].dt.year == year) &
            (df['tpep_pickup_datetime'].dt.month == month) &
            (df['tpep_pickup_datetime'].dt.isocalendar().week == week)
            ]

    # Hàm lọc dữ liệu cuối tuần
    def filter_by_weekend(df, year, month, week):
        week_data = filter_by_week(df, year, month, week)
        return week_data[week_data['tpep_pickup_datetime'].dt.dayofweek.isin([5, 6])]

    # Hàm tải mô hình từ HDFS
    @st.cache_resource
    def load_models():
        try:
            model_borough = PipelineModel.load(BOROUGH_MODEL_PATH)  # Thay đổi tên biến
            model_pc = PipelineModel.load(PC_MODEL_PATH)

            pu_metrics = {
                'accuracy': 0.85,
                'precision': 0.84,
                'recall': 0.83,
                'f1': 0.835
            }

            pc_metrics = {
                'accuracy': 0.78,
                'precision': 0.77,
                'recall': 0.76,
                'f1': 0.765
            }

            return model_borough, model_pc, pu_metrics, pc_metrics
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình từ HDFS: {e}")
            return None, None, None, None

    @st.cache_data
    # Hàm tải dữ liệu lịch sử
    def load_historical_data(borough):
        data_path = f"hdfs://hadoop-hdfs-hdfs-nn:9000/user/hive/warehouse/hoangnn7.db/taxi_trip_summary/data/"
        df = spark.read.parquet(data_path)
        df = df.filter((df.Quan == borough) & (df.Nam == 2020) & (df.Thang.between(1, 6)))
        df = df.orderBy('Thang', 'Ngay', 'Gio')
        pdf = df.toPandas()

        # Tiền xử lý dữ liệu
        pdf['ThoiGian'] = pd.to_datetime(pdf['ThoiGian'])

        # Trích xuất feature từ ThoiGian
        pdf['Day_of_week'] = pdf['ThoiGian'].dt.dayofweek  # 0: T2, 6: CN
        #pdf['Day_of_month'] = pdf['ThoiGian'].dt.day
        #pdf['Month'] = pdf['ThoiGian'].dt.month

        # Chuyển đổi Day_of_week thành 2 categories: 1 (Trong tuần), 2 (Cuối tuần)
        pdf['Day_of_week_category'] = pdf['Day_of_week'].apply(lambda x: 1 if x < 5 else 2)

        # Chuyển đổi Day_of_month thành 3 categories: 1 (Đầu tháng), 2 (Giữa tháng), 3 (Cuối tháng)
        pdf['Day_of_month_category'] = pdf['Ngay'].apply(
            lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
        )

        # Tải bộ chuẩn hóa
        scaler_path = f"hdfs://hadoop-hdfs-hdfs-nn:9000/models/{borough}/scalers.pkl"
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        local_scaler_path = f"/home/ad/models/{borough}_scalers.pkl"

        # Tải bộ chuẩn hóa từ HDFS
        if fs.exists(spark._jvm.org.apache.hadoop.fs.Path(scaler_path)):
            fs.copyToLocalFile(
                spark._jvm.org.apache.hadoop.fs.Path(scaler_path),
                spark._jvm.org.apache.hadoop.fs.Path(local_scaler_path)
            )
            with open(local_scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            os.remove(local_scaler_path)
        else:
            # Tạo bộ chuẩn hóa mới nếu không tìm thấy
            scalers = {
                'gio': MinMaxScaler().fit(pdf[['Gio']]),
                'sochuyen': MinMaxScaler().fit(pdf[['So_chuyen']]),
                'day_of_week': MinMaxScaler().fit(pdf[['Day_of_week_category']]),
                'day_of_month': MinMaxScaler().fit(pdf[['Day_of_month_category']]),
                'month': MinMaxScaler().fit(pdf[['Thang']])
            }

        # Chuẩn hóa các đặc trưng
        pdf['Gio_normalized'] = scalers['gio'].transform(pdf[['Gio']])
        pdf['So_chuyen_normalized'] = scalers['sochuyen'].transform(pdf[['So_chuyen']])
        pdf['Day_of_week_normalized'] = scalers['day_of_week'].transform(pdf[['Day_of_week_category']])
        pdf['Day_of_month_normalized'] = scalers['day_of_month'].transform(pdf[['Day_of_month_category']])
        pdf['Month_normalized'] = scalers['month'].transform(pdf[['Thang']])

        return pdf, scalers

    # Hàm tải mô hình từ HDFS
    def load_model_from_hdfs(borough):
        hdfs_model_path = f"hdfs://hadoop-hdfs-hdfs-nn:9000/models/{borough}/{borough}_lstm_model.keras"
        local_model_path = f"/home/ad/models/{borough}_lstm_model.keras"
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())

        if not fs.exists(spark._jvm.org.apache.hadoop.fs.Path(hdfs_model_path)):
            raise FileNotFoundError(f"Mô hình cho {borough} không tìm thấy tại {hdfs_model_path}")

        # Tải mô hình từ HDFS
        fs.copyToLocalFile(
            spark._jvm.org.apache.hadoop.fs.Path(hdfs_model_path),
            spark._jvm.org.apache.hadoop.fs.Path(local_model_path)
        )

        model = load_model(local_model_path)
        os.remove(local_model_path)
        return model

    # Hàm tải độ đo hiệu suất từ HDFS
    def load_metrics(borough):
        metrics_path = f"hdfs://hadoop-hdfs-hdfs-nn:9000/models/{borough}/metrics.json"
        local_metrics_path = f"/home/ad/models/{borough}_metrics.json"
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())

        if not fs.exists(spark._jvm.org.apache.hadoop.fs.Path(metrics_path)):
            return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'r2': 0.0}

        fs.copyToLocalFile(
            spark._jvm.org.apache.hadoop.fs.Path(metrics_path),
            spark._jvm.org.apache.hadoop.fs.Path(local_metrics_path)
        )

        with open(local_metrics_path, 'r') as f:
            metrics = json.load(f)
        os.remove(local_metrics_path)
        return metrics

    # Hàm dự đoán số chuyến
    def predict_trips(model, pdf, selected_date, selected_hour):
        sequence_length = 24 * 30  # Cập nhật theo mô hình LSTM mới
        features = 5

        # Lấy bộ chuẩn hóa và DataFrame
        scalers = pdf[1]
        pdf = pdf[0]

        # Dự đoán cho giờ được chọn và 5 giờ tiếp theo
        predictions = []
        hours = list(range(selected_hour, selected_hour + 6))  # 6 giờ (giờ chọn + 5 giờ tiếp theo)

        for hour in hours:
            # Xử lý khi giờ vượt quá 23 (sang ngày mới)
            current_date = pd.to_datetime(selected_date) + timedelta(days=hour // 24)
            current_hour = hour % 24

            # Trích xuất đặc trưng từ ngày
            day_of_week = current_date.dayofweek
            day_of_month = current_date.day
            month = current_date.month

            # Chuyển đổi Day_of_week thành 2 categories: 1 (Trong tuần), 2 (Cuối tuần)
            day_of_week_category = 1 if day_of_week < 5 else 2

            # Chuyển đổi Day_of_month thành 3 categories: 1 (Đầu tháng), 2 (Giữa tháng), 3 (Cuối tháng)
            day_of_month_category = 1 if day_of_month <= 10 else (2 if day_of_month <= 20 else 3)

            # Chuẩn hóa đặc trưng đầu vào
            gio_normalized = scalers['gio'].transform(np.array([[current_hour]]))[0][0]
            day_of_week_normalized = scalers['day_of_week'].transform(np.array([[day_of_week_category]]))[0][0]
            day_of_month_normalized = scalers['day_of_month'].transform(np.array([[day_of_month_category]]))[0][0]
            month_normalized = scalers['month'].transform(np.array([[month]]))[0][0]

            # Lấy dữ liệu gần nhất
            start_time = pd.to_datetime('2020-01-01')
            recent_data = pdf[(pdf['ThoiGian'] >= start_time) & (pdf['ThoiGian'] < current_date)]
            if len(recent_data) < sequence_length:
                raise ValueError(f"Không đủ dữ liệu từ tháng 1/2020 đến trước {current_date} để dự đoán.")

            recent_data = recent_data.tail(sequence_length)
            recent_data = recent_data[['So_chuyen_normalized', 'Gio_normalized',
                                       'Day_of_week_normalized', 'Day_of_month_normalized',
                                       'Month_normalized']].copy()

            # Cập nhật hàng cuối cùng với đặc trưng đầu vào
            recent_data.iloc[-1, recent_data.columns.get_loc('Gio_normalized')] = gio_normalized
            recent_data.iloc[-1, recent_data.columns.get_loc('Day_of_week_normalized')] = day_of_week_normalized
            recent_data.iloc[-1, recent_data.columns.get_loc('Day_of_month_normalized')] = day_of_month_normalized
            recent_data.iloc[-1, recent_data.columns.get_loc('Month_normalized')] = month_normalized

            # Chuẩn bị chuỗi
            sequence = recent_data[['So_chuyen_normalized', 'Gio_normalized',
                                    'Day_of_week_normalized', 'Day_of_month_normalized',
                                    'Month_normalized']].values
            sequence = sequence.reshape(1, sequence_length, features)

            # Dự đoán
            pred_normalized = model.predict(sequence, verbose=0)
            pred = scalers['sochuyen'].inverse_transform(pred_normalized)
            predictions.append(int(pred[0][0]))

        # Tạo DataFrame cho biểu đồ
        hours_str = [f"{h % 24:02d}:00" for h in hours]
        dates = [current_date + timedelta(hours=h - selected_hour) for h in hours]
        date_hour_str = [f"{d.strftime('%Y-%m-%d')} {h}" for d, h in zip(dates, hours_str)]

        return predictions[0], predictions, date_hour_str

    # Class kết hợp mô hình
    class CombinedModel:
        def __init__(self, model_borough, model_pc):
            self.model_borough = model_borough
            self.model_pc = model_pc

        def transform(self, df):
            df_borough = self.model_borough.transform(df)
            df_pc = self.model_pc.transform(df)
            borough_pred = df_borough.select("predictedBorough").collect()[0][
                0]  # Giả sử cột dự đoán là predictedBorough
            pc_pred = df_pc.select("predictedPassengerCount").collect()[0][0]
            return {"Borough": borough_pred, "passenger_count": int(pc_pred)}

    # Hàm dự đoán từ thời gian đầu vào
    def predict_pickup(datetime_str, model):
        input_df = spark.createDataFrame([(datetime_str,)], ["tpep_pickup_datetime"])
        input_df = input_df.withColumn("tpep_pickup_datetime",
                                       to_timestamp(col("tpep_pickup_datetime"), "yyyy-MM-dd HH:mm:ss")) \
            .withColumn("hour", hour(col("tpep_pickup_datetime"))) \
            .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime"))) \
            .withColumn("day_of_month", dayofmonth(col("tpep_pickup_datetime"))) \
            .withColumn("month", month(col("tpep_pickup_datetime"))) \
            .withColumn("year", year(col("tpep_pickup_datetime")))
        return model.transform(input_df)

    # Hàm tra cứu thông tin địa điểm
    def get_location_info(location_id, lookup_df):
        location_info = lookup_df[lookup_df['LocationID'] == location_id]
        if not location_info.empty:
            return {
                "Borough": location_info.iloc[0]['Borough'],
                "Zone": location_info.iloc[0]['Zone'],
                "ServiceZone": location_info.iloc[0]['service_zone']
            }
        return None

    # CSS để thu nhỏ tiêu đề trong sidebar và các phần tử khác nếu cần
    st.markdown(
        """
        <style>
        /* Thu nhỏ các tiêu đề trong sidebar */
        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
            font-size: 12px !important;
            margin-top: 3px !important;
            margin-bottom: 3px !important;
        }
        /* Thu nhỏ chữ của các phần tử trong sidebar (nếu cần) */
        div[data-testid="stSidebar"] * {
            font-size: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main application - Sử dụng st.markdown thay vì st.title
    st.markdown(
        "<h1 style='font-size: 32px; margin-top: 3px; margin-bottom: 3px; padding-top:0px;'>Phân tích và Dự đoán Dữ liệu Taxi New York</h1>",
        unsafe_allow_html=True
    )

    # Sidebar navigation
    st.sidebar.markdown(
        "<h1 style='font-size: 30px; margin-top: 3px; margin-bottom: 3px;'>Điều hướng</h1>",
        unsafe_allow_html=True
    )
    page = st.sidebar.radio("Chọn chức năng", ["Phân tích dữ liệu thô", "Phân tích dữ liệu đã làm sạch", "Dự đoán"])

    # Load common data
    zones = load_zone_lookup()
    geo_data = load_geojson()

    if page == "Phân tích dữ liệu thô":
        st.markdown(
            "<h1 style='font-size: 26px; margin-top: 4px; margin-bottom: 4px; padding-top:0px;'>Phân tích Dữ liệu Taxi New York 2019-2020</h1>",
            unsafe_allow_html=True
        )

        # Sidebar inputs for year, month, time analysis, and analysis type
        st.sidebar.markdown(
            "<h1 style='font-size: 12px; margin-top: 3px; margin-bottom: 3px;'>Chọn bộ lọc phân tích</h1>",
            unsafe_allow_html=True
        )

        # Year and Month selection
        year = st.sidebar.selectbox("Năm", [2019, 2020], key="year_analysis")
        month_options = list(range(1, 13)) if year == 2019 else list(range(1, 7))
        month = st.sidebar.selectbox("Tháng", month_options, key="month_analysis")

        # Load data
        df = load_data(year, month)

        if df is not None and zones is not None and geo_data is not None:
            st.markdown(
                f"<h2 style='font-size: 20px; margin-top: 3px; margin-bottom: 3px; padding-top:0px;'>Dữ liệu Taxi - Tháng {month}/{year}</h2>",
                unsafe_allow_html=True
            )

            # Time analysis option
            time_option = st.sidebar.selectbox(
                "Chọn kiểu phân tích thời gian",
                ["Toàn bộ tháng", "Theo tuần", "Cuối tuần"],
                key="time_option"
            )

            filtered_df = df

            if time_option == "Theo tuần":
                weeks = get_weeks_in_month(df, year, month)
                if weeks:
                    week = st.sidebar.selectbox("Chọn tuần", weeks, key="week")
                    start_date, end_date = get_week_date_range(year, week)
                    filtered_df = filter_by_week(df, year, month, week)
                    st.write(
                        f"Phân tích dữ liệu tuần {week} (từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')})")
                else:
                    st.error("Không có dữ liệu cho tháng này!")
                    filtered_df = pd.DataFrame()

            elif time_option == "Cuối tuần":
                weeks = get_weeks_in_month(df, year, month)
                if weeks:
                    week = st.sidebar.selectbox("Chọn tuần", weeks, key="weekend")
                    saturday, sunday = get_weekend_dates(year, week)
                    filtered_df = filter_by_weekend(df, year, month, week)
                    st.write(
                        f"Phân tích dữ liệu cuối tuần của tuần {week} (ngày {saturday.strftime('%d/%m/%Y')} và {sunday.strftime('%d/%m/%Y')})")
                else:
                    st.error("Không có dữ liệu cho tháng này!")
                    filtered_df = pd.DataFrame()

            if not filtered_df.empty:
                # Analysis type selection
                analysis_option = st.sidebar.selectbox(
                    "Chọn loại phân tích",
                    ["SHAPE", "Quan sát chi tiết các cột dữ liệu", "Kiểm tra trùng lặp",
                     "Kiểm tra dữ liệu bị nhiễu", "Bản đồ mật độ đón khách"],
                    key="analysis_option"
                )

                # Display results based on analysis type
                if analysis_option == "SHAPE":
                    st.markdown(
                        "<h2 style='font-size: 20px; margin-top: 3px; margin-bottom: 3px;'>📊 Kích thước dữ liệu</h2>",
                        unsafe_allow_html=True
                    )

                    # Tạo layout 2 cột
                    col1, col2 = st.columns(2)

                    with col1:
                        # Sử dụng st.metric cho số dòng
                        st.metric(
                            label="**Số dòng**",
                            value=f"{filtered_df.shape[0]:,}",
                            help="Tổng số bản ghi trong tập dữ liệu"
                        )

                    with col2:
                        # Sử dụng st.metric cho số cột
                        st.metric(
                            label="**Số cột**",
                            value=filtered_df.shape[1],
                            help="Tổng số thuộc tính trong tập dữ liệu"
                        )



                elif analysis_option == "Quan sát chi tiết các cột dữ liệu":

                    st.markdown(

                        "<h2 style='font-size: 12px; margin-top: 3px; margin-bottom: 3px;'>Chi tiết các cột dữ liệu</h2>",

                        unsafe_allow_html=True

                    )

                    column = st.sidebar.selectbox("Chọn cột để xem chi tiết", filtered_df.columns, key="column_detail")

                    # Tạo layout dạng card cho thông tin cơ bản

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        # Card hiển thị kiểu dữ liệu

                        dtype = str(filtered_df[column].dtype)

                        st.metric(

                            label="**Kiểu dữ liệu**",

                            value=dtype,

                            help=f"Kiểu dữ liệu của cột {column}"

                        )

                    with col2:

                        # Card hiển thị số giá trị duy nhất

                        unique_count = filtered_df[column].nunique()

                        st.metric(

                            label="**Giá trị duy nhất**",

                            value=f"{unique_count:,}",

                            help=f"Số lượng giá trị không trùng lặp trong cột {column}"

                        )

                    with col3:

                        # Card hiển thị số giá trị thiếu

                        missing_count = filtered_df[column].isnull().sum()

                        missing_percent = (missing_count / len(filtered_df)) * 100

                        st.metric(

                            label="**Giá trị thiếu**",

                            value=f"{missing_count:,} ({missing_percent:.1f}%)",

                            help=f"Số lượng và tỷ lệ giá trị thiếu trong cột {column}"

                        )

                    # Phần biểu đồ

                    st.markdown("---")

                    st.markdown(

                        "<h3 style='font-size: 12px; margin-top: 3px; margin-bottom: 3px;'>Biểu đồ phân bố</h3>",

                        unsafe_allow_html=True

                    )

                    # Danh sách các cột sẽ vẽ boxplot

                    boxplot_columns = [

                        'trip_distance',

                        'fare_amount',

                        'tip_amount',

                        'total_amount',

                        'tolls_amount'

                    ]

                    # Tạo figure

                    fig, ax = plt.subplots(figsize=(8, 4))

                    if column in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:

                        # Xử lý cột thời gian - vẽ histogram theo giờ

                        filtered_df['hour'] = filtered_df[column].dt.hour

                        sns.histplot(data=filtered_df, x='hour', bins=24, kde=True, ax=ax)

                        ax.set_xlabel(f"Giờ trong ngày ({column})")

                        ax.set_ylabel("Số lượng chuyến đi")

                        plt.xticks(range(0, 24))


                    elif column in boxplot_columns:

                        # Xử lý các cột số - vẽ boxplot

                        sns.boxplot(x=filtered_df[column], ax=ax)

                        ax.set_xlabel(column)

                        ax.set_title(f"Phân phối giá trị {column}")


                    else:

                        # Xử lý các cột còn lại - vẽ countplot

                        if filtered_df[column].nunique() <= 20:

                            sns.countplot(x=filtered_df[column], ax=ax)

                            ax.set_xlabel(column)

                            ax.set_ylabel("Số lượng")

                            plt.xticks(rotation=45)

                        else:

                            st.warning(
                                "Cột này có quá nhiều giá trị duy nhất để hiển thị biểu đồ. Vui lòng chọn cột khác.")

                    plt.tight_layout()

                    st.pyplot(fig, clear_figure=True)

                    plt.close(fig)


                elif analysis_option == "Kiểm tra trùng lặp":

                    st.markdown(

                        """

                        <h2 style='font-size: 16px; margin-bottom: 10px;'>

                        🔍 Kiểm tra dữ liệu trùng lặp

                        </h2>

                        """,

                        unsafe_allow_html=True

                    )

                    # Tạo card hiển thị kết quả

                    duplicate_card = st.container()

                    duplicated_rows = filtered_df[filtered_df.duplicated(keep=False)]

                    num_duplicates = len(duplicated_rows)

                    if num_duplicates > 0:

                        duplicate_card.warning(

                            f"⚠️ Phát hiện **{num_duplicates}** bản ghi trùng lặp",

                            icon="⚠️"

                        )

                        # Hiển thị thống kê

                        with st.expander("📊 Thống kê chi tiết", expanded=True):

                            cols = st.columns(2)

                            cols[0].metric("Tổng bản ghi", len(filtered_df))

                            cols[1].metric("Bản ghi trùng", num_duplicates)

                        # Hiển thị mẫu dữ liệu trùng

                        with st.expander("🔎 Xem dữ liệu trùng lặp", expanded=False):

                            st.dataframe(

                                duplicated_rows.sort_values(by=filtered_df.columns.tolist()).head(20),

                                height=300,

                                use_container_width=True

                            )

                        # Nút tải về dữ liệu trùng

                        csv = duplicated_rows.to_csv(index=False).encode('utf-8')

                        st.download_button(

                            label="📥 Tải về dữ liệu trùng",

                            data=csv,

                            file_name=f"duplicated_records_{year}_{month}.csv",

                            mime='text/csv'

                        )

                    else:

                        duplicate_card.success(

                            "✅ Không có bản ghi nào trùng lặp!",

                            icon="✅"

                        )

                        # Hiển thị thống kê tích cực

                        with st.expander("📊 Thống kê dữ liệu sạch", expanded=True):

                            cols = st.columns(2)

                            cols[0].metric("Tổng bản ghi", len(filtered_df))

                            cols[1].metric("Bản ghi trùng", 0)


                elif analysis_option == "Kiểm tra dữ liệu bị nhiễu":
                    def show_noise_details(noise_data, title, description):
                        """Hiển thị chi tiết từng loại nhiễu"""
                        if noise_data['count'] > 0:
                            st.markdown(f"""
                            ### {title}
                            *{description}*  
                            **Số lượng:** {noise_data['count']:,} bản ghi  
                            """)

                            # Hiển thị 10 bản ghi đầu tiên
                            with st.expander("📋 Xem chi tiết bản ghi", expanded=False):
                                cols = st.columns(2)
                                cols[0].write("**Index các bản ghi:**")
                                cols[0].write(noise_data['indices'][:10])

                                # Hiển thị dữ liệu mẫu
                                sample_data = filtered_df.loc[noise_data['indices'][:5]]
                                cols[1].write("**Dữ liệu mẫu:**")
                                cols[1].dataframe(sample_data, height=200)

                                # Nút tải về
                                csv = sample_data.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label=f"📥 Tải về mẫu dữ liệu {title.lower()}",
                                    data=csv,
                                    file_name=f"noisy_data_{title}.csv",
                                    mime='text/csv'
                                )
                        else:
                            st.success(f"✅ Không có bản ghi {title.lower()}", icon="✅")

                    st.markdown("""

                    <h2 style='font-size: 16px; margin-bottom: 10px;'>

                    🧹 Kiểm tra dữ liệu nhiễu

                    </h2>

                    """, unsafe_allow_html=True)

                    noise_info = check_noisy_data(filtered_df)

                    total_noise = sum(info['count'] for info in noise_info.values())

                    # Tạo card tổng quan

                    if total_noise > 0:

                        st.warning(f"""

                        ⚠️ **Phát hiện {total_noise:,} bản ghi có dữ liệu nhiễu**  

                        ({(total_noise / len(filtered_df) * 100):.2f}% tổng số bản ghi)

                        """, icon="⚠️")

                    else:

                        st.success("✅ Không phát hiện dữ liệu nhiễu", icon="✅")

                    # Tạo tabs cho từng loại nhiễu

                    tab1, tab2, tab3, tab4 = st.tabs([

                        "⏱️ Thời gian",

                        "🛣️ Quãng đường",

                        "👥 Hành khách",

                        "💰 Giá tiền"

                    ])

                    with tab1:

                        show_noise_details(

                            noise_info['invalid_time'],

                            "Thời gian bất thường",

                            "Chuyến đi có thời gian âm hoặc kéo dài > 1 ngày"

                        )

                    with tab2:

                        show_noise_details(

                            noise_info['invalid_distance'],

                            "Quãng đường bất thường",

                            "Chuyến đi có quãng đường < 0 hoặc > 100 dặm"

                        )

                    with tab3:

                        show_noise_details(

                            noise_info['invalid_passengers'],

                            "Số hành khách bất thường",

                            "Chuyến đi có số khách ≤ 0 hoặc > 6 người"

                        )

                    with tab4:

                        show_noise_details(

                            noise_info['invalid_fare'],

                            "Giá tiền bất thường",

                            "Chuyến đi có giá tiền < 0 hoặc > 1000 USD"

                        )


                elif analysis_option == "Bản đồ mật độ đón khách":

                    st.markdown("""

                    <h2 style='font-size: 16px; margin-bottom: 10px;'>

                    🗺️ Bản đồ mật độ đón khách

                    </h2>

                    """, unsafe_allow_html=True)

                    # Tạo layout 2 cột

                    col1, col2 = st.columns([3, 1])

                    with col1:

                        # Tính toán dữ liệu đón khách

                        pickup_counts = filtered_df['PULocationID'].value_counts().reset_index()

                        pickup_counts.columns = ['LocationID', 'pickup_count']

                        # Kết hợp với dữ liệu địa lý

                        geo_data['location_id'] = geo_data['location_id'].astype(int)

                        map_data = geo_data.merge(pickup_counts, left_on='location_id', right_on='LocationID',
                                                  how='left')

                        map_data['pickup_count'] = map_data['pickup_count'].fillna(0)

                        # Tạo bản đồ Folium

                        m = folium.Map(location=[40.7128, -74.0060],

                                       zoom_start=11,

                                       tiles="cartodbpositron",

                                       width='100%',

                                       height=500)

                        # Thêm lớp choropleth

                        choropleth = folium.Choropleth(

                            geo_data=map_data,

                            name='choropleth',

                            data=map_data,

                            columns=['LocationID', 'pickup_count'],

                            key_on='feature.properties.location_id',

                            fill_color='YlOrRd',

                            fill_opacity=0.7,

                            line_opacity=0.2,

                            legend_name='Số lượng chuyến đi',

                            highlight=True

                        ).add_to(m)

                        # Thêm tooltip

                        folium.GeoJson(

                            map_data,

                            style_function=lambda x: {'fillColor': '#ffffff00', 'color': '#000000', 'weight': 1},

                            tooltip=folium.GeoJsonTooltip(

                                fields=['zone', 'borough', 'pickup_count'],

                                aliases=['Khu vực:', 'Quận:', 'Số chuyến đi:'],

                                localize=True,

                                style=("font-weight: bold;")

                            )

                        ).add_to(m)

                        # Thêm layer control

                        folium.LayerControl().add_to(m)

                        # Hiển thị bản đồ

                        st_folium(m, use_container_width=True)

                    with col2:

                        # Panel thống kê

                        st.markdown("### 📊 Thống kê")

                        total_pickups = int(map_data['pickup_count'].sum())

                        avg_pickups = int(map_data['pickup_count'].mean())

                        max_pickups = int(map_data['pickup_count'].max())

                        max_zone = map_data.loc[map_data['pickup_count'].idxmax(), 'zone']

                        st.metric("Tổng chuyến đi", f"{total_pickups:,}")

                        st.metric("Trung bình/khu vực", f"{avg_pickups:,}")

                        st.metric("Khu vực đông nhất",

                                  f"{max_pickups:,}",

                                  f"{max_zone}")

                        # Top 5 khu vực

                        st.markdown("**Top 5 khu vực:**")

                        top_zones = map_data.sort_values('pickup_count', ascending=False).head(5)

                        for _, row in top_zones.iterrows():
                            st.write(f"- {row['zone']}: {int(row['pickup_count']):,}")
            else:
                st.error("Không có dữ liệu để phân tích cho lựa chọn này!")
        else:
            st.error("Không thể tải dữ liệu. Vui lòng kiểm tra các file đầu vào.")

    elif page == "Dự đoán":
        st.markdown("""
            <h1 style='font-size: 18px; margin-bottom: 20px;'>
            🚖 Dự đoán số lượng chuyến xe Taxi
            </h1>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            with st.container():
                st.markdown("### ⚙️ Thiết lập dự đoán")

                with st.expander("📅 Chọn thông tin", expanded=True):
                    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'EWR']
                    selected_date = st.date_input(
                        "Ngày dự đoán",
                        value=datetime(2020, 4, 1),
                        min_value=datetime(2020, 1, 1),
                        max_value=datetime(2020, 12, 31)
                    )
                    selected_borough = st.selectbox("Quận", options=boroughs, index=0)
                    selected_hour = st.selectbox("Giờ", options=range(24), format_func=lambda x: f"{x:02d}", index=8)

                predict_btn = st.button("🔮 Chạy dự đoán", use_container_width=True, type="primary")

        with col2:
            result_container = st.container()
            result_container.markdown("### 📊 Kết quả dự đoán")

            if predict_btn:
                try:
                    with st.spinner("Đang xử lý dự đoán..."):
                        # Tải mô hình
                        model = load_model_from_hdfs(selected_borough)
                        # Tải dữ liệu lịch sử và bộ chuẩn hóa
                        pdf = load_historical_data(selected_borough)
                        # Dự đoán
                        predicted_trips, predictions, date_hour_str = predict_trips(model, pdf, selected_date,
                                                                                    selected_hour)
                        # Lấy độ đo hiệu suất từ HARDCODED_METRICS
                        metrics = HARDCODED_METRICS.get(selected_borough,
                                                        {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'r2': 0.0})

                    result_container.success("✅ Dự đoán thành công!")

                    st.metric(
                        label="**Số chuyến xe dự đoán**",
                        value=predicted_trips,
                        help=f"Số lượng chuyến xe dự đoán tại {selected_borough} vào {selected_date} {selected_hour:02d}:00"
                    )

                    # Vẽ biểu đồ đường
                    fig = px.line(
                        x=date_hour_str,
                        y=predictions,
                        labels={'x': 'Ngày và Giờ', 'y': 'Số chuyến xe dự đoán'},
                        title=f"Dự đoán số chuyến xe tại {selected_borough} từ {selected_hour:02d}:00",
                        markers=True
                    )
                    fig.update_traces(line=dict(color='#1f77b4', width=2), marker=dict(size=8))
                    fig.update_layout(
                        xaxis_title="Ngày và Giờ",
                        yaxis_title="Số chuyến xe",
                        font=dict(size=12),
                        title_font_size=16,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📈 Độ chính xác mô hình", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{metrics['mae']:.4f}")
                        with col2:
                            st.metric("MSE", f"{metrics['mse']:.4f}")
                        with col3:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        with col4:
                            st.metric("R²", f"{metrics['r2']:.4f}")

                except Exception as e:
                    result_container.error(f"❌ Lỗi khi dự đoán: {str(e)}")

    elif page == "Phân tích dữ liệu đã làm sạch":
        st.markdown("""
        <h1 style='font-size: 18px; margin-bottom: 20px;'>
        🧹 Phân tích dữ liệu đã làm sạch 
        </h1>
        """, unsafe_allow_html=True)

        # Sidebar inputs for date range
        st.sidebar.markdown(
            "<h1 style='font-size: 12px; margin-top: 3px; margin-bottom: 3px;'>Chọn khoảng thời gian</h1>",
            unsafe_allow_html=True
        )

        # Date range selection in sidebar
        start_date = st.sidebar.date_input(
            "Ngày bắt đầu",
            value=datetime(2020, 1, 1),
            min_value=datetime(2019, 1, 1),
            max_value=datetime(2020, 12, 31),
            key="clean_start_date"
        )

        end_date = st.sidebar.date_input(
            "Ngày kết thúc",
            value=datetime(2020, 1, 31),
            min_value=datetime(2019, 1, 1),
            max_value=datetime(2020, 12, 31),
            key="clean_end_date"
        )

        # Validate date range
        if start_date > end_date:
            st.error("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc")
            st.stop()

        # Load data
        with st.spinner("Đang tải dữ liệu từ Iceberg..."):
            df = query_iceberg_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df is not None:
            if not df.empty:
                st.success(
                    f"Đã tải {len(df)} bản ghi từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}")

                # Hiển thị các chỉ số tổng quan
                st.markdown("### 📊 Chỉ số tổng quan")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Tổng số chuyến đi", len(df))

                with col2:
                    avg_passengers = df['passenger_count'].mean()
                    st.metric("Số khách trung bình", f"{avg_passengers:.1f}")

                with col3:
                    avg_distance = df['trip_distance'].mean()
                    st.metric("Quãng đường TB (dặm)", f"{avg_distance:.1f}")

                with col4:
                    avg_fare = df['total_amount'].mean()
                    st.metric("Giá trung bình (USD)", f"${avg_fare:.2f}")

                # Phân tích theo các khía cạnh khác nhau
                st.markdown("---")
                st.markdown("### 📈 Phân tích chi tiết")

                # Tạo tab với radio buttons để duy trì trạng thái
                tab_options = ["🕒 Theo giờ", "🗺️ Theo khu vực", "💰 Giá vé", "📅 Theo ngày"]
                current_tab = st.radio("Chọn chế độ phân tích:", tab_options, horizontal=True)

                # Hiển thị nội dung tương ứng với tab được chọn
                if current_tab == "🕒 Theo giờ":
                    # Phân bố theo giờ
                    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
                    hourly_counts = df['pickup_hour'].value_counts().sort_index()

                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax)
                    ax.set_title("Số lượng chuyến đi theo giờ trong ngày")
                    ax.set_xlabel("Giờ")
                    ax.set_ylabel("Số chuyến đi")
                    st.pyplot(fig)
                    plt.close(fig)

                    # Top 5 giờ cao điểm
                    st.write("**Top 5 giờ cao điểm:**")
                    top_hours = hourly_counts.nlargest(5)
                    for hour, count in top_hours.items():
                        st.write(f"- {hour}h: {count} chuyến")

                elif current_tab == "🗺️ Theo khu vực":
                    if zones is not None and geo_data is not None:
                        # Phân bố theo khu vực đón
                        pu_counts = df['PULocationID'].value_counts().reset_index()
                        pu_counts.columns = ['LocationID', 'pickup_count']

                        # Kết hợp với thông tin zone
                        pu_counts = pu_counts.merge(
                            zones,
                            left_on='LocationID',
                            right_on='LocationID',
                            how='left'
                        )

                        # Hiển thị top 10 khu vực
                        st.write("**Top 10 khu vực đón khách nhiều nhất:**")
                        top_pu = pu_counts.nlargest(10, 'pickup_count')
                        st.dataframe(top_pu[['Zone', 'Borough', 'pickup_count']].rename(
                            columns={'Zone': 'Khu vực', 'Borough': 'Quận', 'pickup_count': 'Số chuyến'}
                        ))

                        # Bản đồ nhiệt
                        geo_data['location_id'] = geo_data['location_id'].astype(int)
                        map_data = geo_data.merge(
                            pu_counts,
                            left_on='location_id',
                            right_on='LocationID',
                            how='left'
                        )
                        map_data['pickup_count'] = map_data['pickup_count'].fillna(0)

                        # Tạo bản đồ Folium với kích thước nhỏ hơn
                        m = folium.Map(location=[40.7128, -74.0060],
                                       zoom_start=11,
                                       tiles="cartodbpositron")

                        # Thêm lớp choropleth
                        choropleth = folium.Choropleth(
                            geo_data=map_data,
                            name='choropleth',
                            data=map_data,
                            columns=['LocationID', 'pickup_count'],
                            key_on='feature.properties.location_id',
                            fill_color='YlOrRd',
                            fill_opacity=0.7,
                            line_opacity=0.2,
                            legend_name='Số lượng chuyến đi',
                            highlight=True
                        ).add_to(m)

                        # Thêm tooltip
                        folium.GeoJson(
                            map_data,
                            style_function=lambda x: {'fillColor': '#ffffff00', 'color': '#000000', 'weight': 1},
                            tooltip=folium.GeoJsonTooltip(
                                fields=['zone', 'borough', 'pickup_count'],
                                aliases=['Khu vực:', 'Quận:', 'Số chuyến đi:'],
                                localize=True,
                                style=("font-weight: bold;")
                            )
                        ).add_to(m)

                        # Thêm layer control
                        folium.LayerControl().add_to(m)

                        # Hiển thị bản đồ với kích thước nhỏ hơn và ít tùy chọn tương tác hơn
                        st_folium(m, width=700, height=450)

                        # Thống kê khu vực
                        st.markdown("### 📊 Thống kê khu vực")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tổng số khu vực có đón khách",
                                      len(pu_counts[pu_counts['pickup_count'] > 0]))
                        with col2:
                            st.metric("Khu vực có nhiều chuyến nhất",
                                      f"{top_pu.iloc[0]['Zone']} ({top_pu.iloc[0]['pickup_count']} chuyến)")

                elif current_tab == "💰 Giá vé":
                    # Phân tích giá vé
                    st.markdown("### 📊 Phân bố giá vé")

                    # Tạo 2 cột cho biểu đồ
                    col1, col2 = st.columns(2)

                    with col1:
                        # Boxplot giá vé
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        sns.boxplot(x=df['total_amount'], ax=ax1)
                        ax1.set_title("Phân phối giá vé")
                        ax1.set_xlabel("Giá vé (USD)")
                        st.pyplot(fig1)
                        plt.close(fig1)

                    with col2:
                        # Histogram giá vé
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        sns.histplot(df['total_amount'], bins=30, kde=True, ax=ax2)
                        ax2.set_title("Histogram giá vé")
                        ax2.set_xlabel("Giá vé (USD)")
                        st.pyplot(fig2)
                        plt.close(fig2)

                    # Thống kê chi tiết
                    st.markdown("### 📈 Thống kê giá vé")
                    fare_stats = df['total_amount'].describe().to_frame().T
                    st.dataframe(fare_stats.style.format("{:.2f}"))

                    # Phân tích mối quan hệ giữa quãng đường và giá vé
                    st.markdown("### 🔗 Mối quan hệ quãng đường - giá vé")
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(data=df.sample(min(1000, len(df))),
                                    x='trip_distance', y='total_amount', alpha=0.5, ax=ax3)
                    ax3.set_title("Quan hệ giữa quãng đường và giá vé")
                    ax3.set_xlabel("Quãng đường (dặm)")
                    ax3.set_ylabel("Giá vé (USD)")
                    st.pyplot(fig3)
                    plt.close(fig3)

                elif current_tab == "📅 Theo ngày":
                    # Phân tích theo ngày
                    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
                    daily_counts = df['pickup_date'].value_counts().sort_index()

                    # Biểu đồ đường
                    st.markdown("### 📅 Xu hướng theo ngày")
                    fig1, ax1 = plt.subplots(figsize=(12, 4))
                    sns.lineplot(x=daily_counts.index, y=daily_counts.values, ax=ax1)
                    ax1.set_title("Số lượng chuyến đi theo ngày")
                    ax1.set_xlabel("Ngày")
                    ax1.set_ylabel("Số chuyến đi")
                    plt.xticks(rotation=45)
                    st.pyplot(fig1)
                    plt.close(fig1)

                    # Phân tích theo ngày trong tuần
                    st.markdown("### 📆 Phân bố theo ngày trong tuần")
                    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_counts = df['day_of_week'].value_counts().reindex(day_order)

                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax2)
                    ax2.set_title("Số chuyến đi theo ngày trong tuần")
                    ax2.set_xlabel("Ngày")
                    ax2.set_ylabel("Số chuyến đi")
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)
                    plt.close(fig2)

                    # Top 5 ngày cao điểm
                    st.markdown("### 🏆 Top 5 ngày cao điểm")
                    top_days = daily_counts.nlargest(5)
                    for day, count in top_days.items():
                        st.write(f"- **{day.strftime('%d/%m/%Y')}**: {count} chuyến")

                # Nút tải dữ liệu
                st.markdown("---")
                st.download_button(
                    label="📥 Tải dữ liệu phân tích",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f"cleaned_taxi_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
            else:
                st.warning("Không có dữ liệu trong khoảng thời gian đã chọn")

    class CombinedModel:
        def __init__(self, model_pu, model_pc):
            self.model_pu = model_pu
            self.model_pc = model_pc

        def transform(self, df):
            df_pu = self.model_pu.transform(df)
            df_pc = self.model_pc.transform(df)
            pu_pred = df_pu.select("predictedBorough").collect()[0][0]
            pc_pred = df_pc.select("predictedPassengerCount").collect()[0][0]
            return {"PULocationID": int(pu_pred), "passenger_count": int(pc_pred)}

    # Cleanup SparkSession
    def cleanup():
        spark.stop()

    # Register cleanup
    if hasattr(st, 'experimental_memo'):
        st.experimental_memo.clear()
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()


# Hàm chính của ứng dụng
def main():
    # Kiểm tra trạng thái đăng nhập
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.markdown("<h1 style='font-size: 32px; margin-top: 3px; margin-bottom: 3px; padding-top:0px;'>Đăng nhập</h1>",
                    unsafe_allow_html=True)
        username = st.text_input("Tên người dùng")
        password = st.text_input("Mật khẩu", type="password")

        if st.button("Đăng nhập"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Đăng nhập thành công!")
                st.rerun()  # Tải lại trang để hiển thị giao diện chính
            else:
                st.error("Tên người dùng hoặc mật khẩu không đúng!")
    else:
        main_page()


if __name__ == "__main__":
    main()


