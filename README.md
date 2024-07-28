# Using Apache Pinot as a Kafka Consumer and Datastore for Fast On-the-Fly Aggregations

This is a PoC I prepared to demonstrate positioning Apache Pinot as a Kafka consumer and datastore to enable fast on-the-fly aggregations.

[Here](https://mert.codes/harder-better-faster-stronger-apache-pinot-as-a-kafka-consumer-and-datastore-for-fast-7df25bcc7d02) is my post about this PoC.

### The Ingredients

- Producer
- Message broker
- Destination (consumer)

### Technologies Used

- Kafka
- Pinot
- Python

### To Run

_**Note:** If you get `Docker Container exited with code 137` errors during running and experimenting, chances are that your Docker Desktop -> Resources -> Memory amount is not enough. I have set my resource settings as follows: CPUs 6, Memory 8 GB, Swap 1 GB._
`docker build -t real_time_detect .`
Open up a terminal and browse to the cloned folder and execute the following command to see the magic happen:

`docker-compose up`

Or to have everything run at the background silently, add -d

`docker-compose up -d`

Open another terminal and execute a script to create table in Apache Pinot:

`./run.sh`

Browse Apache Pinot UI to query some data

http://localhost:9000/#/query

Execute the following SQL repeatedly to see the sample on-the-fly aggregation results:
> SELECT person_name, sum(incoming_amount) FROM poc
GROUP BY person_name

![image](https://github.com/user-attachments/assets/8c775526-df57-46fb-9d2d-826867fe765a)
