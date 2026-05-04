# face-recognition-opencv

Módulo de visão computacional responsável pela detecção e reconhecimento facial em tempo real. Este projeto atua como o cliente (edge) do sistema de autenticação biométrica.

---

## 🔬 Tecnologias e Algoritmos

- **Python 3**: Linguagem base para o processamento de dados.
- **OpenCV**: Biblioteca principal para manipulação de vídeo e extração de características faciais.
- **LBPH (Local Binary Patterns Histograms)**: Algoritmo de reconhecimento facial utilizado pela sua alta eficiência em hardware limitado e robustez contra variações de iluminação.
- **Haar Cascades**: Utilizado para a detecção rápida de faces em cada frame da captura.

---

## 🛠️ Funcionalidades

- **Treinamento Global**: Processamento de múltiplas classes de usuários com normalização de imagem (150x150 pixels) para garantir uniformidade nos histogramas.
- **Negative Sampling**: Inclusão de classes de "Desconhecidos" para reduzir drasticamente a taxa de falsos positivos e aumentar a confiança do modelo.
- **Processamento de Imagem**: Aplicação de `equalizeHist` para normalizar o contraste e lidar com diferentes condições de luz ambiente.
- **Histórico de Acurácia**: Utilização de uma fila (`deque`) para calcular a média de confiança dos últimos 20 frames, evitando oscilações visuais no reconhecimento.

---

## 🔄 Integração com o Backend

Este módulo coleta os dados e os envia via requisições HTTP POST para a API de autenticação.

**Repositório do Backend (Go)**: [facial-authentication-api](https://github.com/JRafaelRosa/facial-authentication-api)

---

## 📂 Estrutura de Pastas
```text
.
├── app/
│   ├── entities/       # Definição do modelo Person
│   ├── model/          # Lógica de visão (train_global e recognition)
│   └── services/       # Cliente HTTP para envio de dados
├── data/
│   ├── dataset/        # Imagens organizadas por pastas (nomes dos usuários)
│   └── train/          # Modelos exportados (.yml) e mapas de nomes (.pkl)
└── main.py             # Orquestrador do sistema
```
### 🚀 Como Utilizar
**Prepare o Dataset:

Coloque 10 a 20 fotos suas em data/dataset/SeuNome/.

Adicione fotos de outras pessoas em data/dataset/Desconhecidos/ para criar contraste.

**Instale as Dependências:
```
pip install -r requirements.txt
```
**Inicie o Reconhecimento:
```
python main.py
```
Pressione 'S' durante a execução para capturar frames e atualizar o treinamento global automaticamente.

Pressione 'Q' para sair.
---
Desenvolvido por Joao Rafael dos Santos da Rosa. Focado em otimização de modelos de visão computacional e integração de sistemas distribuídos.
