from locust import TaskSet, task, between, User, HttpUser
import lorem
import json

class WebsiteUser(HttpUser):
    # @task is used for the locust undestand that we want to request the API in the function

    #wait_time = between(1, 3)
    #min_wait = 5000
    #max_wait = 15000

    @task
    def postagging(self):
        self.client.post(
            url="http://localhost:8000/predict",
            data=json.dumps({"sentence": lorem.sentence(), "sentence_compare": lorem.sentence()}),
            name="http://localhost:8000/predict")