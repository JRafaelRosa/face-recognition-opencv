# face-recognition-opencv

Módulo de visão computacional responsável pela detecção e reconhecimento facial em tempo real. Este projeto atua como o cliente (edge) do sistema de autenticação biométrica.

---

## ⚠️ Status do Projeto

Este projeto encontra-se **em construção ativa**.

O sistema já realiza detecção e reconhecimento facial funcional, porém a **precisão do modelo está sendo continuamente refinada**, com ajustes em:

- melhoria da taxa de acerto (accuracy)
- redução de falsos positivos em ambientes com baixa iluminação
- estabilização do reconhecimento entre frames consecutivos
- otimização do dataset e balanceamento de classes

---

## 🔬 Tecnologias e Algoritmos

- **Python 3**: Linguagem base para o processamento de dados.
- **OpenCV**: Biblioteca principal para manipulação de vídeo e extração de características faciais.
- **LBPH (Local Binary Patterns Histograms)**: Algoritmo de reconhecimento facial utilizado pela sua eficiência em hardware limitado e robustez contra variações de iluminação.
- **Haar Cascades**: Utilizado para detecção rápida de faces em cada frame da captura.

---

## 🛠️ Funcionalidades

- **Treinamento Global**: Processamento de múltiplas classes de usuários com normalização de imagem (150x150 pixels).
- **Negative Sampling**: Inclusão da classe "Desconhecido" para reduzir falsos positivos e aumentar a confiabilidade do modelo.
- **Processamento de Imagem**: Uso de `equalizeHist` para normalização de contraste em diferentes condições de luz.
- **Histórico de Acurácia**: Uso de `deque` para média dos últimos 20 frames, reduzindo oscilações no reconhecimento.

---

## 🔄 Integração com o Backend

Este módulo coleta os dados e envia via requisições HTTP POST para a API de autenticação.

**Repositório do Backend (Go):**  
https://github.com/JRafaelRosa/facial-authentication-api

---

## 📂 Estrutura de Pastas

```text
.
├── app/
│   ├── entities/       # Modelo Person
│   ├── model/          # Lógica de treino e reconhecimento
│   └── services/       # Cliente HTTP para envio de dados
├── data/
│   ├── dataset/        # Imagens por usuário
│   └── train/          # Modelos treinados (.yml, .pkl)
└── main.py             # Orquestrador do sistema
```

## 🏗️ Arquitetura
```
Camera
   ↓
OpenCV + LBPH
   ↓
Validação de confiança
   ↓
HTTP POST
   ↓
API Go
   ↓
MySQL
```

## 🚀 Como Utilizar
**Preparar o Dataset

Coloque imagens do usuário em:
```
data/dataset/SeuNome/
```

Adicione imagens de controle em:
```
data/dataset/Desconhecidos/
```

**Instale as Dependências:
```
pip install -r requirements.txt
```
**Executar o sistema
```
python main.py
```
Pressione 'S' durante a execução para capturar frames e atualizar o treinamento global automaticamente.

Pressione 'Q' para sair.

## 🧠 Objetivo
Este projeto tem como foco:

- aprimorar técnicas de visão computacional aplicada
- estudar reconhecimento facial com LBPH
- integração de sistemas distribuídos (Python → API Go)
- otimização de precisão em cenários reais simulados

## 📌 Próximos Passos

- migração para embeddings faciais
- testes com FaceNet/DeepFace
- melhoria do pipeline de inferência
- suporte a múltiplas câmeras
  
---
Desenvolvido por Joao Rafael dos Santos da Rosa. Focado em otimização de modelos de visão computacional e integração de sistemas distribuídos.
