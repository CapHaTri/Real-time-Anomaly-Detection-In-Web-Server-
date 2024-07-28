import pandas as pd
from confluent_kafka import Producer, KafkaError
import simplejson as json
import time
import pymysql
import sys

# Kafka producer configuration
producer_conf = {
    'bootstrap.servers': 'kafka:9092',
    'linger.ms': 10
}

producer = Producer(producer_conf)

def delivery_report(err, msg):
    """ Callback called on successful or failed delivery of message """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def read_mysql_and_produce(db_opts, sql, topic_name):
    db = None
    try:
        # Connect to MySQL
        db = pymysql.connect(**db_opts)
        cur = db.cursor()

        # Execute the query
        cur.execute(sql)
        rows = cur.fetchall()
        if rows:
            column_names = [desc[0] for desc in cur.description]

            # Iterate through rows and produce to Kafka
            for row in rows:
                message = {column_names[i]: row[i] for i in range(len(column_names))}
                message['value'] = float(message['value'])  # Ensure value is properly formatted as float
                print(message)
                producer.produce(topic_name, value=json.dumps(message), callback=delivery_report)
                producer.poll(1)  # Serve delivery reports (callbacks)
                time.sleep(1)  # Sleep for a second before sending the next message
        else:
            sys.exit(f"No rows found for query: {sql}")

    except Exception as exc:
        print(f"Error while reading MySQL and producing to Kafka: {exc}")

    finally:
        if db is not None:
            db.close()
        producer.flush()
        print("Producer flushed, shutting down...")

# Database configuration
db_opts = {
    'user': 'root',
    'password': 'sql',
    'host': 'mysql',  # MySQL service name defined in Docker Compose
    'database': 'time_series',
    'port': 3306
}

# SQL query and Kafka topic name
sql_query = 'SELECT * from time_series'
kafka_topic = 'poc.topic'

# Read data from MySQL and produce to Kafka
read_mysql_and_produce(db_opts, sql_query, kafka_topic)
