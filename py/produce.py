import pandas as pd
from confluent_kafka import Producer, KafkaError
import simplejson as json
import time

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

def read_csv_and_produce(file_path, topic_name):
    try:
        # Read CSV file into pandas DataFrame
        df = pd.read_csv(file_path)

        # Iterate through rows in DataFrame
        for index, row in df.iterrows():
            timestamp = row['timestamp']
            value = row['value']

            # Prepare message to send to Kafka
            message = {
                'timestamp': timestamp,
                'value': value
            }
            print(message)
            # Produce message to Kafka
            producer.produce(topic_name, value=json.dumps(message), callback=delivery_report)
            producer.poll(1)  # Serve delivery reports (callbacks)
            time.sleep(1)  # Sleep for a second before sending the next message

    except Exception as exc:
        print(f"Error while reading CSV and producing to Kafka: {exc}")

    finally:
        producer.flush()
        print("Producer flushed, shutting down...")

# Đường dẫn đến file CSV và tên topic Kafka
csv_file_path = '/home/ec2_cpu_utilization_77c1ca.csv'
kafka_topic = 'poc.topic'

# Gọi hàm để đọc từ CSV và gửi dữ liệu vào Kafka
read_csv_and_produce(csv_file_path, kafka_topic)

# from confluent_kafka import Producer, KafkaError
# from confluent_kafka.admin import AdminClient, NewTopic
# import simplejson as json
# import random
# import time

# def delivery_report(err, msg):
#     """ Callback called on successful or failed delivery of message """
#     if err is not None:
#         print(f"Message delivery failed: {err}")
#     else:
#         print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

# try:
#     print('Hello. I am about to publish messages to Kafka...')

#     # Admin client to create topic
#     admin_client = AdminClient({
#         "bootstrap.servers": "kafka:9092"
#     })
    
#     topic_list = [NewTopic("poc.topic", num_partitions=1, replication_factor=1)]
    
#     # Create topic if it doesn't exist
#     fs = admin_client.create_topics(topic_list)

#     for topic, f in fs.items():
#         try:
#             f.result()  # The result itself is None
#             print(f"Topic {topic} created")
#         except Exception as e:
#             print(f"Failed to create topic {topic}: {e}")

#     # Kafka producer configuration
#     producer_conf = {
#         'bootstrap.servers': 'kafka:9092',
#         'linger.ms': 10
#     }

#     producer = Producer(producer_conf)
    
#     people = [
#         {"name": "Jane", "surname": "Doe", "bankAccountId": "ABC110"},
#         {"name": "Joe", "surname": "Doe", "bankAccountId": "DEF220"},
#         {"name": "Bonnie", "surname": "Brown", "bankAccountId": "GHI330"},
#         {"name": "Clyde", "surname": "Crimson", "bankAccountId": "JKL440"}
#     ]

#     id_counter = 1

#     while True:
#         try:
#             person = random.choice(people)
#             person["incomingAmount"] = round(random.uniform(10, 50), 2)
#             person["incomingId"] = id_counter
#             producer.produce('poc.topic', value=json.dumps(person), callback=delivery_report)
#             producer.poll(1)  # Serve delivery reports (callbacks)
#             id_counter += 1
#             time.sleep(1)  # Sleep for a second before sending the next message
#         except KeyboardInterrupt:
#             break
#         except Exception as exc:
#             print(f"Error: {exc}")
# finally:
#     producer.flush()
#     print("Shutting down producer.")
# try:
#     print('Hello. I am about to publish messages to Kafka...')
#     try:
#         admin_client = AdminClient({
#             "bootstrap.servers": "kafka:9092"
#         })
#         topic_list = []
#         topic_list.append(NewTopic("poc.topic", 1, 1))
#         admin_client.create_topics(topic_list)
#     except Exception as ex:
#         print(ex)
#         pass
#     producer_conf = {
#         'bootstrap.servers': 'kafka:9092',
#         'linger.ms': 10
#     }

#     producer = Producer(producer_conf)
#     people = [
#         {"name": "Jane", "surname": "Doe", "bankAccountId": "ABC110"},
#         {"name": "Joe", "surname": "Doe", "bankAccountId": "DEF220"},
#         {"name": "Bonnie", "surname": "Brown", "bankAccountId": "GHI330"},
#         {"name": "Clyde", "surname": "Crimson", "bankAccountId": "JKL440"},
#         {"name": "Bonnie", "surname": "Brown", "bankAccountId": "GHI330"},
#         {"name": "Clyde", "surname": "Crimson", "bankAccountId": "JKL440"},
#         {"name": "Bonnie", "surname": "Brown", "bankAccountId": "GHI330"},
#         {"name": "Clyde", "surname": "Crimson", "bankAccountId": "JKL440"}
#     ]
#     id_counter = 1
#     while True:
#         try:
#             person = random.choice(people)
#             person["incomingAmount"] = round(random.uniform(10,50), 2)
#             person["incomingId"] = id_counter
#             producer.produce('poc.topic', value=json.dumps(person), callback=delivery_report)
#             producer.flush()
#             print('Sent message with incomingId: {}', id_counter)
#             id_counter = id_counter + 1
#             time.sleep(5)
#         except Exception as exc:
#             print(exc)
# except Exception as e:
#     print(e)
#     pass