import cv2 as cv
import numpy as np
import pickle
import os
import time
from collections import deque
from app.entities.Person import Person

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train(person: Person):
    path_dataset = os.path.join(BASE_DIR, 'data', 'dataset')
    path_train = os.path.join(BASE_DIR, 'data', 'train')

    os.makedirs(path_dataset, exist_ok=True)
    os.makedirs(path_train, exist_ok=True)

    face_cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector_treino = cv.CascadeClassifier(face_cascade_path)

    faces = []
    ids = []
    reconhecedor_lbph = cv.face.LBPHFaceRecognizer_create()
    id_usuario = abs(hash(person.name)) % 10000

    path_person = os.path.join(path_dataset, person.name)
    if not os.path.exists(path_person):
        os.makedirs(path_person, exist_ok=True)
        return False

    fotos_existentes = [f for f in os.listdir(path_person) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not fotos_existentes:
        return False

    for photo in fotos_existentes:
        caminho_img = os.path.join(path_person, photo)
        img = cv.imread(caminho_img)
        if img is None: continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        coordenadas = detector_treino.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in coordenadas:
            faces.append(gray[y:y + h, x:x + w])
            ids.append(id_usuario)

    if not faces: return False

    reconhecedor_lbph.train(faces, np.array(ids))

    path_yml = os.path.join(path_train, f"{person.name}.yml")
    path_pkl = os.path.join(path_train, f"{person.name}_dado.pkl")
    reconhecedor_lbph.save(path_yml)

    dataPerson = {
        'id': id_usuario, 'path': os.path.abspath(path_yml),
        'name': person.name, 'email': person.email,
        'position': person.position, 'accuracy': 0.0
    }

    with open(path_pkl, 'wb') as f:
        pickle.dump(dataPerson, f)

    print(f"--- Modelo de {person.name} atualizado com sucesso! ---")
    return True


def recognition(active_person: Person):
    path_train = os.path.join(BASE_DIR, 'data', 'train')
    path_dataset = os.path.join(BASE_DIR, 'data', 'dataset', active_person.name)
    os.makedirs(path_dataset, exist_ok=True)
    os.makedirs(path_train, exist_ok=True)

    reconhecedores = {}
    pessoas = {}

    def carregar_modelos():
        nonlocal reconhecedores, pessoas
        if not os.path.exists(path_train): return
        for arquivo in os.listdir(path_train):
            if arquivo.endswith('.pkl'):
                try:
                    with open(os.path.join(path_train, arquivo), 'rb') as f:
                        dados = pickle.load(f)
                    rec = cv.face.LBPHFaceRecognizer_create()
                    rec.read(dados['path'])
                    reconhecedores[dados['id']] = rec
                    pessoas[dados['id']] = dados
                except:
                    pass

    carregar_modelos()
    detector_camera = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv.VideoCapture(0)

    # Variáveis de controle para coleta dinâmica
    coletando = False
    amostras_coletadas = 0
    LIMITE_AMOSTRAS = 25
    historico = {pid: deque(maxlen=20) for pid in pessoas}

    print(f"Logado como: {active_person.name}")
    print("Comandos: [S] Coletar novo grupo de frames e Treinar | [Q] Sair")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector_camera.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            rosto_roi = gray[y:y + h, x:x + w]
            rosto_roi_eq = cv.equalizeHist(rosto_roi)

            # --- MODO COLETA E TREINO AUTOMÁTICO ---
            if coletando:
                amostras_coletadas += 1
                img_name = os.path.join(path_dataset, f"auto_{int(time.time())}_{amostras_coletadas}.jpg")
                cv.imwrite(img_name, rosto_roi)  # Salva o frame bruto para o dataset

                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv.putText(frame, f"Coletando: {amostras_coletadas}/{LIMITE_AMOSTRAS}", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if amostras_coletadas >= LIMITE_AMOSTRAS:
                    coletando = False
                    amostras_coletadas = 0
                    print("Iniciando treinamento automático...")
                    if train(active_person):
                        carregar_modelos()  # Recarrega o modelo atualizado em memória

            # --- MODO RECONHECIMENTO ---
            else:
                melhor_id, melhor_conf = None, float('inf')
                for pid, rec in reconhecedores.items():
                    label, conf = rec.predict(rosto_roi_eq)
                    if conf < melhor_conf:
                        melhor_conf, melhor_id = conf, pid

                if melhor_id and melhor_conf < 110:
                    historico.setdefault(melhor_id, deque(maxlen=20)).append(melhor_conf)
                    acc = max(0, 100 - (sum(historico[melhor_id]) / len(historico[melhor_id])))
                    txt, color = f"{pessoas[melhor_id]['name']} {acc:.1f}%", (0, 255, 0)
                else:
                    txt, color = "Desconhecido", (0, 0, 255)

                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, txt, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv.imshow("Sistema Inteligente (S para Treinar)", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s') and not coletando:
            coletando = True
            print(f"Coletando {LIMITE_AMOSTRAS} frames para {active_person.name}...")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Exemplo: Define quem o sistema deve treinar ao pressionar 'S'
    usuario_atual = Person(name="Joao", email="joao@email.com", position="Engenheiro")
    recognition(usuario_atual)