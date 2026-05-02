import json
import requests
from app.entities.Person import Person
from app.model.validator import json_data

def send_data(url_api, person: Person):
    data = json_data(person)
    try:
        response = requests.post(url_api, json=data)
        if response.status_code == 201:
            print(f"Sucesso {response.text}")
        else:
            print(f"Erro {response.status_code}: {response.text}")
            return
    except Exception as e:
        print(f"Erro {e}")

