import cv2 as cv
import numpy as np
import pickle
import os
import time
from collections import deque
from app.entities.Person import Person

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import cv2 as cv
import numpy as np
import os
import pickle


def train():
    """
    Treina o modelo com todas as pastas presentes em data/dataset.
    Isso cria o contraste necessário entre 'Você' e 'Desconhecidos'.
    """
    path_dataset = os.path.join(BASE_DIR, 'data', 'dataset')
    path_train = os.path.join(BASE_DIR, 'data', 'train')

    os.makedirs(path_train, exist_ok=True)

    face_cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector_treino = cv.CascadeClassifier(face_cascade_path)

    faces = []
    ids = []
    mapa_nomes = {}  # Para guardar qual ID pertence a qual nome

    reconhecedor_lbph = cv.face.LBPHFaceRecognizer_create()

    # Percorre cada pasta de pessoa (ex: joao_rosa, desconhecidos)
    for nome_pasta in os.listdir(path_dataset):
        path_person = os.path.join(path_dataset, nome_pasta)

        if not os.path.isdir(path_person):
            continue

        # Gera um ID único para cada pasta/pessoa
        id_atual = abs(hash(nome_pasta)) % 10000
        mapa_nomes[id_atual] = nome_pasta

        print(f"Processando fotos de: {nome_pasta} (ID: {id_atual})")

        fotos = [f for f in os.listdir(path_person) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for photo in fotos:
            caminho_img = os.path.join(path_person, photo)
            img = cv.imread(caminho_img)
            if img is None: continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.equalizeHist(gray)  # Mantendo sua padronização de luz

            # Detecta o rosto na foto do dataset
            coordenadas = detector_treino.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in coordenadas:
                # O segredo da acurácia: Redimensionar para um tamanho padrão
                rosto_recortado = cv.resize(gray[y:y + h, x:x + w], (150, 150))
                faces.append(rosto_recortado)
                ids.append(id_atual)

    if not faces:
        print("Nenhuma face encontrada para treinar.")
        return False

    # Treina o modelo com TODOS os rostos e IDs de uma vez
    reconhecedor_lbph.train(faces, np.array(ids))

    # Salva o modelo único que contém todas as pessoas
    path_yml = os.path.join(path_train, "modelo_global.yml")
    reconhecedor_lbph.save(path_yml)

    # Salva o mapa de IDs para você saber quem é quem no recognition
    path_mapa = os.path.join(path_train, "mapa_nomes.pkl")
    with open(path_mapa, 'wb') as f:
        pickle.dump(mapa_nomes, f)

    print(f"--- Treino Global concluído! {len(mapa_nomes)} classes mapeadas ---")
    return True


import cv2 as cv
import numpy as np
import pickle
import os
import time
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_global():
    """
    Varre a pasta dataset, treina um modelo LBPH com todas as pessoas
    e gera o contraste necessário para subir a acurácia.
    """
    path_dataset = os.path.join(BASE_DIR, 'data', 'dataset')
    path_train = os.path.join(BASE_DIR, 'data', 'train')
    os.makedirs(path_train, exist_ok=True)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    reconhecedor_lbph = cv.face.LBPHFaceRecognizer_create()

    faces, ids = [], []
    mapa_nomes = {}

    for nome_pasta in os.listdir(path_dataset):
        path_person = os.path.join(path_dataset, nome_pasta)
        if not os.path.isdir(path_person): continue

        id_atual = abs(hash(nome_pasta)) % 10000
        mapa_nomes[id_atual] = nome_pasta

        print(f"Treinando: {nome_pasta} (ID: {id_atual})")

        for foto in os.listdir(path_person):
            if not foto.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            img = cv.imread(os.path.join(path_person, foto))
            if img is None: continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.equalizeHist(gray)

            deteccoes = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in deteccoes:
                # O SEGREDO: Redimensionar para um tamanho padrão fixo
                rosto_fatiado = cv.resize(gray[y:y + h, x:x + w], (150, 150))
                faces.append(rosto_fatiado)
                ids.append(id_atual)

    if not faces: return False

    reconhecedor_lbph.train(faces, np.array(ids))

    # Salva o modelo único e o mapa de nomes
    reconhecedor_lbph.save(os.path.join(path_train, "modelo_global.yml"))
    with open(os.path.join(path_train, "mapa_nomes.pkl"), 'wb') as f:
        pickle.dump(mapa_nomes, f)

    print("--- Modelo Global Atualizado ---")
    return True


def recognition(active_person):
    path_train = os.path.join(BASE_DIR, 'data', 'train')
    path_dataset_pessoal = os.path.join(BASE_DIR, 'data', 'dataset', active_person.name)
    os.makedirs(path_dataset_pessoal, exist_ok=True)

    modelo_global = cv.face.LBPHFaceRecognizer_create()
    mapa_nomes = {}

    def carregar_recursos():
        nonlocal mapa_nomes
        try:
            modelo_global.read(os.path.join(path_train, "modelo_global.yml"))
            with open(os.path.join(path_train, "mapa_nomes.pkl"), 'rb') as f:
                mapa_nomes = pickle.load(f)
            return True
        except:
            return False

    carregar_recursos()
    detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Se o Win der erro de índice, tente VideoCapture(0, cv.CAP_DSHOW)
    cap = cv.VideoCapture(0)

    coletando = False
    amostras = 0
    LIMITE = 25
    historico = {pid: deque(maxlen=20) for pid in mapa_nomes}

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            rosto_roi = gray[y:y + h, x:x + w]
            rosto_roi_eq = cv.equalizeHist(rosto_roi)

            if coletando:
                amostras += 1
                cv.imwrite(os.path.join(path_dataset_pessoal, f"cap_{int(time.time())}_{amostras}.jpg"), rosto_roi)

                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                if amostras >= LIMITE:
                    coletando, amostras = False, 0
                    if train_global(): carregar_recursos()
            else:
                # Otimização: Redimensiona uma vez antes do predict
                rosto_pronto = cv.resize(rosto_roi_eq, (150, 150))
                id_previsto, conf = modelo_global.predict(rosto_pronto)

                # No LBPH, menor confiança = maior precisão.
                # Com classe negativa, 85-90 é um ótimo threshold.
                if id_previsto in mapa_nomes and conf < 90:
                    historico.setdefault(id_previsto, deque(maxlen=20)).append(conf)
                    acc = max(0, 100 - (sum(historico[id_previsto]) / len(historico[id_previsto])))
                    nome = mapa_nomes[id_previsto]
                    txt, color = f"{nome} {acc:.1f}%", (0, 255, 0)
                else:
                    txt, color = "Desconhecido", (0, 0, 255)

                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, txt, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv.imshow("Reconhecimento (S p/ Treinar | Q p/ Sair)", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s') and not coletando:
            coletando = True
            print("Coletando novas amostras...")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Exemplo: Define quem o sistema deve treinar ao pressionar 'S'
    usuario_atual = Person(name="Joao", email="joao@email.com", position="Engenheiro")
    recognition(usuario_atual)
    #train()