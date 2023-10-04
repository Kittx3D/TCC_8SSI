import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def execucao(imagem):
    def blob_imagem(net, imagem, mostrar_texto=True):
        # tempo exato logo antes de iniciar o processamento da imagem e predição
        inicio = time.time()

        # constroi o blob da imagem de entrada
        blob = cv2.dnn.blobFromImage(
            imagem, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # parâmetros:
        # - imagem de entrada, que queremos pré-processar antes de passar para a rede
        # - fator de escala, tamanho que será redimensionado
        # - tamanho que a rede neural espera
        # - swapRB: deixamos =True para inverter os canais RGB (pois OpenCV trabalha com BGR)
        # - crop: controla se parte da imagem será cortada para encaixar no tamanho.
        #         Se =False não irá cortar, mantendo a proporção

        # passa o blob da imagem como entrada da rede
        net.setInput(blob)

        # realiza a predição e obtem os resultados
        layerOutputs = net.forward(ln)

        # tempo exato logo após terminar o processamento
        termino = time.time()

        return net, imagem, layerOutputs

    def deteccoes(detection, _threshold, caixas, confiancas, IDclasses, mostrar_texto=True):

        scores = detection[5:]
        # ID da classe da detecção com maior score (o mais provável)
        classeID = np.argmax(scores)
        confianca = scores[classeID]  # retorna a confiança, ao acessar o valor

        if confianca > _threshold:
            
            caixa = detection[0:4] * np.array([W, H, W, H])

            (centerX, centerY, width, height) = caixa.astype("int")

            # usa as coordenadas (x, y) do centro para encontrar o topo e o canto esquerdo da caixa de seleção
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # atualiza a lista de coordenadas da caixa de detecção, confianças e IDs de classes
            caixas.append([x, y, int(width), int(height)])
            confiancas.append(float(confianca))
            IDclasses.append(classeID)

        return caixas, confiancas, IDclasses

    def funcoes_imagem(imagem, i, confiancas, caixas, COLORS, LABELS, mostrar_texto=True):

        # extrai as coordenadas da caixa de detecção
        (x, y) = (caixas[i][0], caixas[i][1])
        (w, h) = (caixas[i][2], caixas[i][3])

        # define uma cor única para a classe
        cor = [int(c) for c in COLORS[IDclasses[i]]]

        # desenha o retângulo (caixa de detecção) ao redor do objeto, com base nas coordenadas extraídas
        cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2)
        
        percentual_confianca = confiancas[i] * 100

        # vamos colocar a confiança da predição ao lado do nome da classe
        texto = "{}: {:.1f}%".format(LABELS[IDclasses[i]], percentual_confianca)

        if mostrar_texto:
            print("> " + texto)
            print(x, y, w, h)

        # escreve acima do retângulo o label (nome) do objeto e a confiança
        cv2.putText(imagem, texto, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        return imagem, x, y, w, h

    caixas = []
    confiancas = []
    IDclasses = []

    _threshold = 0.5
    _threshold_NMS = 0.3

    labels_path = os.path.sep.join(['D:/Serv/API_TCC/content/cfg', 'obj.names'])
    LABELS = open(labels_path).read().strip().split("\n")

    config_path = os.path.sep.join(['D:/Serv/API_TCC/content/cfg', 'yolov4_custom.cfg'])
    weights_path = os.path.sep.join(['D:/Serv/API_TCC/content', 'yolov4_custom_best.weights'])

    net = cv2.dnn.readNet(config_path, weights_path)

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    ln = net.getLayerNames()

    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    (H, W) = imagem.shape[:2]

    net, imagem, layerOutputs = blob_imagem(net, imagem)

    for output in layerOutputs:
        for detection in output:
            caixas, confiancas, IDclasses = deteccoes(
                detection, _threshold, caixas, confiancas, IDclasses)

    objs = cv2.dnn.NMSBoxes(caixas, confiancas, _threshold, _threshold_NMS)

    if len(objs) > 0:

        for i in objs.flatten():

            imagem, x, y, w, h = funcoes_imagem(
                imagem, i, confiancas, caixas, COLORS, LABELS, mostrar_texto=False)
            objeto = imagem[y:y + h, x:x + w]

    # Salva a imagem processada em um arquivo temporário
    processed_image_path = f'D:/Serv/API_TCC/uploads/{nome_imagem}.jpg'
    cv2.imwrite(processed_image_path, imagem)
    
    return processed_image_path

# processar imagem
@app.route('/wildfire/processar', methods=['POST'])
def enviar_imagem():
    if 'arquivo_imagem' not in request.files:
        return jsonify({'Erro': 'Nenhuma imagem informada!'})
    
    imagem = request.files['arquivo_imagem']
    
    if imagem and imagem.filename.endswith('.jpg'):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            
        nome_imagem = os.path.join(app.config['UPLOAD_FOLDER'], imagem.filename)
        imagem.save(nome_imagem)
        
        # Carregue a imagem usando OpenCV
        imagem = cv2.imread(nome_imagem)

        # Chame a função execucao com a imagem carregada
        processed_image_path = execucao(imagem)
    
        return mostrar(processed_image_path)
    
    return jsonify({'Erro': 'Arquivo Inválido!'})

# Mostrar Imagem:
@app.route('/wildfire/mostrar/<path:processed_image_path>', methods=['GET'])
def mostrar(processed_image_path):
    return send_file(processed_image_path, mimetype='image/jpeg')

# Mostrar Opções
@app.route('/wildfire', methods=['GET'])
def opcoes():
    opcoes_funcao = [
        {
            'Rotas': '/wildfire  -> Menu'
        },
        {
            'Rotas': '/wildfire/processar  -> Processar imagem'
        },
        {
            'Rotas': '/wildfire/mostrar/<path:processed_image_path>  -> Mostrar Imagem'
        },
    ]

    return jsonify(opcoes_funcao)

# rodando
if __name__ == '__main__':
    app.run(port=5000, host='localhost', debug=True)
