FROM bitnami/airflow:2.10.1

USER root
RUN apt update && apt install -y default-jdk

COPY env.sh /opt/env.sh
RUN mkdir -p /opt/hadoop /opt/spark /opt/bitnami/airflow/code
# COPY trino /opt/trino
# RUN chmod +x /opt/trino
# COPY hadoop /opt
# COPY spark /opt

# Đổi quyền sở hữu và cấp quyền giống /tmp
RUN chown -R root:1001 /opt/spark /opt/hadoop && chmod -R 2777 /opt/spark /opt/hadoop
# Cấp quyền cho tất cả file con trong /opt/bitnami/airflow giống như /opt/spark
RUN find /opt/bitnami/airflow -path /opt/bitnami/airflow/licenses -prune -o -path /opt/bitnami/airflow/logs -prune -o -path /opt/bitnami/airflow/nss-wrapper -prune -o -path /opt/bitnami/airflow/tmp -prune -o -path /opt/bitnami/airflow/venv -prune -o -exec chown root:1001 {} \; -exec chmod 2777 {} \;
# Cài đặt thư viện cần thiết
RUN pip install pyspark confluent_kafka

# USER 1001