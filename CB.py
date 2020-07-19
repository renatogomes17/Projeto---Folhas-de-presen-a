import pyzbar.pyzbar as pyzbar

def processaCodigoBarras(img):
    try:
        decodeCB = pyzbar.decode(img)
    except:
        quit("Erro ao tentar ler o Código de Barras, Folha ilegível")

    #retira o último digito do código de barras por não ser necessário.
    if len(decodeCB) > 0:
        codigo = decodeCB[0].data
        codigo = int(codigo)
        codigo = int(codigo / 10)
    else:
        quit("Erro ao tentar ler o Código de Barras, Folha ilegível")

    return codigo