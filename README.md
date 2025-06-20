# IT4997 - Bachelor Thesis - SOICT- HUST

## Introduction
<ul>
  <li>Name of project: Building a data collection and processing system to support transportation demand analysis using components of the Hadoop ecosystem</li>
  <li>Project objective:
    <ul>
      <li>Explore Hadoop ecosystem components in big data processing storage with Data Lake architecture</li>
      <li>Mastering Hadoop system administration for Taxi data analysis</li>
      <li>Monitor performance, fault tolerance, load balancing, and information security issues</li>
    </ul>
  </li>
</ul>

## System architecture
  <img src="https://github.com/hoangnm1111/Hadoop_Ecosystem_With_K8s/blob/main/img/system.png">

## Deploy
### 1. Install tools
#### 1.1 Install Kubernetes
```
https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/
```

#### 1.2 Install Helm
```
https://helm.sh/docs/intro/install/
```

#### 1.3 Install Lens
```
https://spacelift.io/blog/lens-kubernetes
```

### 2. Create a Kubernetes cluster with Minikube
#### 2.1 Create a cluster
```sh
minikube start --nodes 3 -p hadoop-ecosystem
```

#### 2.2 Label nodes
```sh
kubectl label node hadoop-ecosystem-m02 node-role.kubernetes.io/worker=worker & kubectl label nodes hadoop-ecosystem-m02 role=worker
```
```sh
kubectl label node hadoop-ecosystem-m03 node-role.kubernetes.io/worker=worker & kubectl label nodes hadoop-ecosystem-m03 role=worker
```

### 3. Deploy system
#### 3.0 Create namespace
```sh
kubectl create namespace hadoop-ecosystem & kubectl config set-context --current --namespace=hadoop-ecosystem
```

#### 3.1 Deploy Flask
```sh
kubectl create -f ./kubernetes/flask
```

#### 3.2 Deploy Kafka
```sh
kubectl create -f ./kubernetes/kafka
```

#### 3.3 Deploy Airflow
```sh
helm install airflow ./kubernetes/airflow
```

#### 3.4 Deploy Hadoop
```sh
helm install hadoop ./kubernetes/hadoop
```

#### 3.5 Deploy Hive
```sh
helm install hive-metastore ./kubernetes/hive-metastore
```

#### 3.6 Deploy Trino
```sh
helm install trino ./kubernetes/trino
```

#### 3.7 Deploy Superset
```sh
helm install superset ./kubernetes/superset
```

### 4. Pre-use
#### 4.1 Download data source
```
https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019
```

#### 4.2 Move data source to system
```sh
kubectl cp /path/to/datasource <pod-flask-1>:/data & kubectl cp /path/to/datasource <pod-flask-2>:/data
```

## Demo
### System overview
  <img src="https://github.com/Tran-Ngoc-Bao/Hadoop_Ecosystem_With_K8s/blob/master/pictures/report/overview.png">
  
### Data flow
  <img src="https://github.com/Tran-Ngoc-Bao/Hadoop_Ecosystem_With_K8s/blob/master/pictures/result/airflow_result_1.png">
  
### Demo output
  <img src="https://github.com/Tran-Ngoc-Bao/Hadoop_Ecosystem_With_K8s/blob/master/pictures/charts/s%E1%BB%91-chuy%E1%BA%BFn-bay-theo-thang-qua-cac-nam-2024-12-01T13-13-34.441Z.jpg">
  
## Report
<ul>
  <li><a href="https://github.com/Tran-Ngoc-Bao/Hadoop_Ecosystem_With_K8s/blob/master/report/report.pdf">Report</a></li>
  <li><a href="https://github.com/Tran-Ngoc-Bao/Hadoop_Ecosystem_With_K8s/blob/master/report/slide.pptx">Slide</a></li>
  <li><a href="https://github.com/Tran-Ngoc-Bao/Hadoop_Ecosystem_With_K8s/blob/master/report/demo.mp4">Demo</a></li>
</ul>
