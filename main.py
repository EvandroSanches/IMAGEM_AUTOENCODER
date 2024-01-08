import matplotlib.pyplot as plt
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, UpSampling2D, Flatten, Reshape, Input
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np


epochs = 50
batch_size = 256

def CarregaDados():
    #Carregando dados
    (previsores_treino, _), (previsores_teste, _) = mnist.load_data()

    #Estruturando dados (60000, 28, 28, 1) - (n, dimensão, dimensão, canal de cor)
    previsores_treino = previsores_treino.reshape((previsores_treino.shape[0], previsores_treino.shape[1],previsores_treino.shape[2], 1))
    previsores_teste = previsores_teste.reshape((len(previsores_teste), previsores_teste.shape[1], previsores_teste.shape[2], 1))

    #Normalizando dados
    previsores_treino = previsores_treino.astype('float32') / 255
    previsores_teste = previsores_teste.astype('float32') / 255

    return previsores_treino, previsores_teste

def CriaRede():
    #Normalização de dados
    modelo = Sequential()

    #Primeira camada encoder
    modelo.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2)))

    #Segunda camada encoder
    modelo.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2), padding='same'))

    #Saida encoder 3, 3, 8
    modelo.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)))
    modelo.add(Flatten())

    #Primeira camada decoder
    modelo.add(Reshape((4,4,8)))
    modelo.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same' ))
    modelo.add(UpSampling2D((2,2)))

    #Segunda camada decoder
    modelo.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    modelo.add(BatchNormalization())
    modelo.add(UpSampling2D((2,2)))

    #Primeira camada encoder
    modelo.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(UpSampling2D((2,2)))

    #Saida decoder 28, 28, 1
    modelo.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))

    #print(modelo.summary()) #- Vizualizar shape das camadas

    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return modelo

def Treinamento():
    #Carregando dados e rede neural
    previsores_treino, previsores_teste = CarregaDados()
    modelo = CriaRede()

    #Definindo callbacks
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=1e-10)
    rlp = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
    md = ModelCheckpoint(filepath='Modelo.0.1', verbose=1, save_best_only=True)

    #Treinando modelo
    result = modelo.fit(previsores_treino, previsores_treino, batch_size=batch_size, epochs=epochs, callbacks=[es, rlp, md], validation_data=(previsores_teste, previsores_teste))

    #Visualizando resultados do treinamento
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('Função de Perda')
    plt.legend(('Treinamento', 'Teste'))
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()

    plt.plot(result.history['accuracy'])
    plt.plot(result.history['val_accuracy'])
    plt.title('Acuracia')
    plt.legend(('Treinamento', 'Teste'))
    plt.xlabel('Épocas')
    plt.ylabel('Acuracia')
    plt.show()

def Encoder(dados):
    #Carregando modelo
    modelo = False

    while modelo == False:
        try:
            modelo = load_model('Modelo.0.1')
        except:
            Treinamento()

    #Definindo camadas do encoder
    encoder = Model(inputs=modelo.input, outputs=modelo.get_layer('flatten').output)

    dados = np.expand_dims(dados, axis=0)

    #Codificando imagem
    imagem_codificada = encoder.predict(dados)

    return imagem_codificada

def Decoder(dados):
    #Carregando modelo
    modelo = False

    while modelo == False:
        try:
            modelo = load_model('Modelo.0.1')
        except:
            Treinamento()

    #Configurando camadas do decoder
    entrada = Input(128)
    camada_decoder1 = modelo.layers[8]
    camada_decoder2 = modelo.layers[9]
    camada_decoder3 = modelo.layers[10]
    camada_decoder4 = modelo.layers[11]
    camada_decoder5 = modelo.layers[12]
    camada_decoder6 = modelo.layers[13]
    camada_decoder7 = modelo.layers[14]
    camada_decoder8 = modelo.layers[15]
    camada_decoder9 = modelo.layers[16]
    camada_decoder10 = modelo.layers[17]
    camada_decoder = camada_decoder10(camada_decoder9(camada_decoder8(camada_decoder7(camada_decoder6(camada_decoder5(camada_decoder4(camada_decoder3(camada_decoder2(camada_decoder1(entrada))))))))))


    #Definindo camadas do decoder
    decoder = Model(inputs=entrada, outputs=camada_decoder)

    #Decodificando imagens
    imagem_decodificada = decoder.predict(dados)

    return imagem_decodificada


previsores_treino, previsores_teste = CarregaDados()

#Gerando indices aleatórios para 10 imagens
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)
plt.Figure(figsize=(18,18))

#Visualizando imagens originais, codificadas e decodificadas
for i, indice_imagem in enumerate(imagens_teste):

    imagem_codificada = Encoder(previsores_teste[indice_imagem])
    imagem_decodificada = Decoder(imagem_codificada)

    eixo = plt.subplot(10,10,i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())

    eixo = plt.subplot(10,10,i + 1 + numero_imagens)
    plt.imshow(imagem_codificada.reshape(16,8))
    plt.xticks(())
    plt.yticks(())

    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagem_decodificada.reshape(28,28))
    plt.xticks(())
    plt.yticks(())

plt.show()
