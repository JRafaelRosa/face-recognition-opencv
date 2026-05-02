import json

class Person:
    def __init__(self, name, email, position):
        self.name = name
        self.email = email
        self.position = position
        self.status = "Waiting"
        self.accuracy = 0.0
