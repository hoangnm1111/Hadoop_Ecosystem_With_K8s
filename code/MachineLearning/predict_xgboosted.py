import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
import os
import pickle
import json

warnings.filterwarnings('ignore')

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("TaxiPickupPredictionManhattanXGBoost") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Đọc dữ liệu từ HDFS và lọc cho Manhattan, tháng 1-6/2020
df = spark.read.parquet("hdfs://hadoop-hdfs-hdfs-nn:9000/user/hive/warehouse/hoangnn7.db/taxi_trip_summary/data/")
df = df.filter((F.col("Nam") == 2020) &
               (F.col("Quan") == "Manhattan"))
df = df.orderBy('Thang', 'Ngay', 'Gio')
pdf = df.toPandas()

# Sử dụng cột ThoiGian (định dạng YYYY-MM-DD) để trích xuất feature
pdf['ThoiGian'] = pd.to_datetime(pdf['ThoiGian'])

# Trích xuất feature từ ThoiGian
pdf['Day_of_week'] = pdf['ThoiGian'].dt.dayofweek  # 0: T2, 6: CN
pdf['Day_of_month_category'] = pdf['Ngay'].apply(
    lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
)
pdf['Day_of_week_category'] = pdf['Day_of_week'].apply(lambda x: 1 if x < 5 else 2)

# Chuẩn hóa các feature
scaler_gio = MinMaxScaler()
scaler_sochuyen = MinMaxScaler()
scaler_day_of_week = MinMaxScaler()
scaler_day_of_month = MinMaxScaler()
scaler_month = MinMaxScaler()

pdf['Gio_normalized'] = scaler_gio.fit_transform(pdf[['Gio']])
pdf['So_chuyen_normalized'] = scaler_sochuyen.fit_transform(pdf[['So_chuyen']])
pdf['Day_of_week_normalized'] = scaler_day_of_week.fit_transform(pdf[['Day_of_week_category']])
pdf['Day_of_month_normalized'] = scaler_day_of_month.fit_transform(pdf[['Day_of_month_category']])
pdf['Month_normalized'] = scaler_month.fit_transform(pdf[['Thang']])

# Lưu bộ chuẩn hóa vào HDFS
scalers = {
    'gio': scaler_gio,
    'sochuyen': scaler_sochuyen,
    'day_of_week': scaler_day_of_week,
    'day_of_month': scaler_day_of_month,
    'month': scaler_month
}
local_scaler_path = "/home/ad/models/Manhattan_xgboost_scalers.pkl"
with open(local_scaler_path, 'wb') as f:
    pickle.dump(scalers, f)

hdfs_scaler_path = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/Manhattan/xgboost_scalers.pkl"
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
hdfs = spark._jvm.org.apache.hadoop.fs.Path(hdfs_scaler_path)

if fs.exists(hdfs):
    fs.delete(hdfs, False)

parent_hdfs = spark._jvm.org.apache.hadoop.fs.Path(os.path.dirname(hdfs_scaler_path))
if not fs.exists(parent_hdfs):
    fs.mkdirs(parent_hdfs)

fs.copyFromLocalFile(
    spark._jvm.org.apache.hadoop.fs.Path(local_scaler_path),
    hdfs
)
os.remove(local_scaler_path)
print(f"Bộ chuẩn hóa đã được lưu vào {hdfs_scaler_path}")

# Chuẩn bị dữ liệu cho XGBoost
data = pdf[['ThoiGian', 'Gio_normalized', 'So_chuyen_normalized',
            'Day_of_week_normalized', 'Day_of_month_normalized', 'Month_normalized']]


# Tạo dữ liệu với lag features thay vì sequences
def create_lag_features(data, lag_hours=24 * 30):
    df = data.copy()
    for i in range(1, lag_hours + 1):
        df[f'So_chuyen_lag_{i}'] = df['So_chuyen_normalized'].shift(i)
    df = df.dropna()
    return df


lag_data = create_lag_features(data, lag_hours=24 * 30)
features = ['Gio_normalized', 'Day_of_week_normalized', 'Day_of_month_normalized', 'Month_normalized'] + \
           [f'So_chuyen_lag_{i}' for i in range(1, 24 * 30 + 1)]
X = lag_data[features]
y = lag_data['So_chuyen_normalized']

# Chia dữ liệu train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Xây dựng mô hình XGBoost
model = XGBRegressor(
    n_estimators=15,
    max_depth=3,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

# Huấn luyện mô hình
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Đánh giá mô hình
y_pred = model.predict(X_test)
y_test_original = scaler_sochuyen.inverse_transform(y_test.values.reshape(-1, 1))
y_pred_original = scaler_sochuyen.inverse_transform(y_pred.reshape(-1, 1))

# Tính các độ đo
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')

# Tạo DataFrame chứa true_value và predict_value
test_time = lag_data['ThoiGian'].iloc[train_size:].reset_index(drop=True)
results_df = pd.DataFrame({
    'Thang': test_time.dt.month,
    'Ngay': test_time.dt.day,
    'Gio': pdf['Gio'].iloc[train_size + 24 * 30:].reset_index(drop=True),
    'true_value': y_test_original.flatten(),
    'predict_value': y_pred_original.flatten()
})

# Lưu DataFrame thành file CSV cục bộ trước
local_csv_path = "/home/ad/PycharmProjects/DA/code/spark/Manhattan_xgboost_predictions.csv"
results_df.to_csv(local_csv_path, index=False)

# Lưu độ đo vào HDFS
metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
local_metrics_path = "/home/ad/models/Manhattan_xgboost_metrics.json"
with open(local_metrics_path, 'w') as f:
    json.dump(metrics, f)

hdfs_metrics_path = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/Manhattan/xgboost_metrics.json"
hdfs = spark._jvm.org.apache.hadoop.fs.Path(hdfs_metrics_path)

if fs.exists(hdfs):
    fs.delete(hdfs, False)

parent_hdfs = spark._jvm.org.apache.hadoop.fs.Path(os.path.dirname(hdfs_metrics_path))
if not fs.exists(parent_hdfs):
    fs.mkdirs(parent_hdfs)

fs.copyFromLocalFile(
    spark._jvm.org.apache.hadoop.fs.Path(local_metrics_path),
    hdfs
)
os.remove(local_metrics_path)
print(f"Độ đo hiệu suất đã được lưu vào {hdfs_metrics_path}")

# Lưu mô hình vào tệp
local_model_path = "/home/ad/models/Manhattan_xgboost_model.pkl"
with open(local_model_path, 'wb') as f:
    pickle.dump(model, f)

# Sao chép mô hình lên HDFS
hdfs_model_path = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/Manhattan/Manhattan_xgboost_model.pkl"
hdfs = spark._jvm.org.apache.hadoop.fs.Path(hdfs_model_path)

if fs.exists(hdfs):
    fs.delete(hdfs, False)

parent_hdfs = spark._jvm.org.apache.hadoop.fs.Path(os.path.dirname(hdfs_model_path))
if not fs.exists(parent_hdfs):
    fs.mkdirs(parent_hdfs)

fs.copyFromLocalFile(
    spark._jvm.org.apache.hadoop.fs.Path(local_model_path),
    hdfs
)
print(f"Mô hình đã được lưu vào {hdfs_model_path}")


# Hàm dự đoán
def predict_trips(ngay, gio):
    ngay = pd.to_datetime(ngay)
    day_of_week = ngay.dayofweek
    day_of_month = ngay.day
    month = ngay.month
    day_of_week_category = 1 if day_of_week < 5 else 2
    day_of_month_category = 1 if day_of_month <= 10 else (2 if day_of_month <= 20 else 3)

    gio_normalized = scaler_gio.transform(np.array([[gio]]))[0][0]
    day_of_week_normalized = scaler_day_of_week.transform(np.array([[day_of_week_category]]))[0][0]
    day_of_month_normalized = scaler_day_of_month.transform(np.array([[day_of_month_category]]))[0][0]
    month_normalized = scaler_month.transform(np.array([[month]]))[0][0]

    start_time = pd.to_datetime('2020-01-01')
    recent_data = pdf[(pdf['ThoiGian'] >= start_time) & (pdf['ThoiGian'] < ngay)]
    if len(recent_data) < 24 * 30:
        raise ValueError(f"Không đủ dữ liệu từ tháng 1/2020 đến trước {ngay} để dự đoán.")

    recent_data = recent_data.tail(24 * 30)
    lag_features = []
    for i in range(1, 24 * 30 + 1):
        lag_features.append(recent_data['So_chuyen_normalized'].iloc[-i])

    input_data = np.array(
        [gio_normalized, day_of_week_normalized, day_of_month_normalized, month_normalized] + lag_features)
    input_data = input_data.reshape(1, -1)

    pred_normalized = model.predict(input_data)
    pred = scaler_sochuyen.inverse_transform(pred_normalized.reshape(-1, 1))
    return pred[0][0]


# Ví dụ sử dụng
try:
    ngay = '2020-07-01'
    gio = 20
    predicted_trips = predict_trips(ngay, gio)
    print(f"Dự đoán số chuyến đi vào {ngay}, Manhattan, {gio}h: {int(predicted_trips)} chuyến")
except ValueError as e:
    print(e)

# Xóa tệp tạm cục bộ
os.remove(local_model_path)

# Đóng SparkSession
spark.stop()
