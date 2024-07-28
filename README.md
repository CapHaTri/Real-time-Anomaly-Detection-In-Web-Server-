# Real time Anomaly Detection In Web Server

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ccede93-c275-4e7a-892f-95b7a617223b" alt="description" />
</p>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Data Pipeline](#data-pipeline)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Conclusion](#concly)
- [Future Direction](#Future_Direction)

## Introduction
This project focuses on detecting anomalies in web server logs in real-time. By leveraging time series data stored in MySQL, Kafka for data streaming, Apache Pinot for real-time analytics, and a custom anomaly detection model, we aim to identify unusual patterns in web server activity. The results are visualized using Streamlit for an interactive demonstration.
## Features
- **Real-time Data Ingestion**: Stream web server logs using Kafka.
- **Efficient Storage:** Use Apache Pinot for storing and querying time series data.
- **Anomaly Detection:** Implement a machine learning model to predict anomalies.
- **Interactive Visualization:** Display results and insights using Streamlit.
- **Docker Compose Setup**: The project uses Docker Compose to streamline the deployment and management of the required services, including Apache Kafka, Apache Pinot, Streamlit, MySQL
## Technologies Used
- **MySQL:** For storing time series data.
- **Kafka:** For real-time data streaming.
- **Apache Pinot:** For real-time data analytics. (Real-time DB)
- **AER model :** For Anomaly Detection (This model be trained)
- **Streamlit:** For creating interactive visualizations.
## Setup Instructions
1. **Clone the Repository**

   ```bash
   git clone https://github.com/CapHaTri/Real-time-Anomaly-Detection-In-Web-Server-.git
   cd Real-time-Anomaly-Detection-In-Web-Server-
2. **Configuration**

- **Model AER**

  - Model availble at model1.pt

- **Producer Configuration**

  - Edit produce.py to set up Producer
- **Streamlit Configuration**
  - Edit app.py (Model inference, Web display)
- **Apache Pinot Configuration**
  - Edit file in Pinot folder to suitable with your data
  - The inference to Streamlit at app.py
- **Docker Compose Configuration**

  - Modify `docker-compose.yml` if necessary to suit your environment.

3.  **Start the Environment**
- Build Image: 
  `docker build -t real_time_detect .`
- Create Topic:
  `docker exec -it kafka kafka-topics.sh --create --topic poc.topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1`
- Open up a terminal and browse to the cloned folder and execute the following command to see the magic happen:
  `docker-compose up`
- Open another terminal and execute a script to create table in Apache Pinot:
  `./run.sh`
## Data Pipeline
![image](https://github.com/user-attachments/assets/e698d8dd-0b4b-48cd-8729-8c16e6591a28)

- **MySQL**: Stores time series data.
- **Docker**: Deploys the services and components of the system.
- **Apache ZooKeeper**: Manages and coordinates Kafka services.
- **Apache Kafka**: Transmits data from MySQL to Apache Pinot.
- **Apache Pinot**: Stores and queries data rapidly.
- **Python**: Executes the anomaly detection model (AER Model).
- **Streamlit**: Visualizes results from the anomaly detection model.

## Usage
### Browse Apache Pinot UI to query some data
- Open ([http://localhost:9000/#/query](http://localhost:9000/#/query)) in your browser.
- Execute the following SQL repeatedly to see the sample on-the-fly aggregation results:
> SELECT * FROM poc
![image](https://github.com/user-attachments/assets/8c775526-df57-46fb-9d2d-826867fe765a)
### Monitor message is deliveried in Kafka Producer
![image](https://github.com/user-attachments/assets/8f2d80c5-4f41-4841-8afe-bf7410d65edd)
### Open Streamlit UI
- Open ([http://localhost:8501](http://localhost:8501)) in your browser.
## Visualizations
![image](https://github.com/user-attachments/assets/1bfab6d7-028f-4ad5-8ce8-a9a0ee1a1732)

- Demo available at : ([https://drive.google.com/file/d/1yXt4FX4fi5-li7gEWJ1ADvEKIyX6yp_E/view?usp=sharing](https://drive.google.com/file/d/1yXt4FX4fi5-li7gEWJ1ADvEKIyX6yp_E/view?usp=sharing))
## Conclusion
- This project provides a robust solution for real-time anomaly detection in web servers, integrating multiple technologies to ensure efficient data processing and insightful visualizations. By continuously monitoring web server logs, it helps in identifying and addressing issues promptly.
## Future Direction
- Enhance Model Accuracy: Improve the anomaly detection model with more advanced techniques.
- Scalability: Optimize the pipeline for handling larger volumes of data.
- Additional Visualizations: Develop more comprehensive visualizations for deeper insights.
- Alerting Mechanism: Implement real-time alerts for detected anomalies.
## Contact
For any questions or feedback, please reach out to [trihx2003@gmail.com](mailto:trihx2003@gmail.com).
