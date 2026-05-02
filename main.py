from app.entities.Person import Person
from app.services.api_client import send_data
from app.model.vision import recognition

# url = "http://localhost:8080/registrar"

def main():
    # train_all()
    person = Person("joao", "teste@test.com", 'estagio')

    data = recognition(person)

    if data:
        print(data)
    else:
        print("Nenhum dado retornado")


if __name__ == "__main__":
    main()