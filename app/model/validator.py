from app.entities.Person import Person

def json_data(person: Person):
    return {
        "Person": {
            "name": person.name,
            "email": person.email,
            "position": person.position
        },
        "status": person.status,
        "accuracy": person.accuracy
    }