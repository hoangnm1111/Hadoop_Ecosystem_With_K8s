import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor  # Thêm Random Forest
import warnings
import os
import pickle
import json

warnings.filterwarnings('ignore')

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("TaxiPickupPredictionManhattan") \
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
pdf['Day_of_week'] = pdf['ThoiGian'].dt.dayofweek
pdf['Day_of_week_category'] = pdf['Day_of_week'].apply(lambda x: 1 if x < 5 else 2)
pdf['Day_of_month_category'] = pdf['Ngay'].apply(
    lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
)

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
local_scaler_path = "/home/ad/models/Manhattan_scalers_rf.pkl"
with open(local_scaler_path, 'wb') as f:
    pickle.dump(scalers, f)

hdfs_scaler_path = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/Manhattan/scalers_rf.pkl"
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

# Kết hợp dữ liệu
data = pdf[['ThoiGian', 'Gio_normalized', 'So_chuyen_normalized',
            'Day_of_week_normalized', 'Day_of_month_normalized', 'Month_normalized']]

# **Sửa đổi: Không cần tạo chuỗi thời gian cho Random Forest**
# Random Forest sử dụng dữ liệu dạng bảng, không cần sequence như LSTM
X = data[['Gio_normalized', 'Day_of_week_normalized', 'Day_of_month_normalized', 'Month_normalized']].values
y = data['So_chuyen_normalized'].values

# Chia dữ liệu train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Xây dựng mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
y_test_original = scaler_sochuyen.inverse_transform(y_test.reshape(-1, 1))
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
test_time = data['ThoiGian'].iloc[train_size:].reset_index(drop=True)
results_df = pd.DataFrame({
    'Thang': test_time.dt.month,
    'Ngay': test_time.dt.day,
    'Gio': pdf['Gio'].iloc[train_size:].reset_index(drop=True),
    'true_value': y_test_original.flatten(),
    'predict_value': y_pred_original.flatten()
})

# Lưu DataFrame thành file CSV cục bộ trước
local_csv_path = "/home/ad/PycharmProjects/DA/code/spark/Manhattan_rf_predictions.csv"
results_df.to_csv(local_csv_path, index=False)

# Lưu độ đo vào HDFS
metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
local_metrics_path = "/home/ad/models/Manhattan_rf_metrics.json"
with open(local_metrics_path, 'w') as f:
    json.dump(metrics, f)

hdfs_metrics_path = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/Manhattan/rf_metrics.json"
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

# Lưu mô hình vào tệp .pkl cục bộ
local_model_path = "/home/ad/models/Manhattan_rf_model.pkl"
with open(local_model_path, 'wb') as f:
    pickle.dump(model, f)

# Sao chép mô hình lên HDFS
hdfs_model_path = "hdfs://hadoop-hdfs-hdfs-nn:9000/models/Manhattan/Manhattan_rf_model.pkl"
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

    # Tạo input cho Random Forest
    input_data = np.array([[gio_normalized, day_of_week_normalized, day_of_month_normalized, month_normalized]])
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
