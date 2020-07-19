import cv2
import utils
import imutils
import numpy as np
import folhaPresenca
import folhaFinal as ff

IMGX = utils.IMGX
IMGY = utils.IMGY
NUM_SAMPLES= []

def carregaNumerosSamples():
    #carrega as imagens dos números que vão servir de base para identificar os nr dos alunos
    #e guarda dentro de um array
    try:
        for i in range(10):
            numeros = cv2.imread("numeros/num" + str(i) + ".JPG")
            numGray = cv2.cvtColor(numeros, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(numGray, 150, 255, cv2.THRESH_BINARY_INV)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            (x, y, w, h) = cv2.boundingRect(contours[0])
            roi = numGray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (60, 70))
            NUM_SAMPLES.append(roi)
    except:
        print("Erro ao carregar os numeros de sample \n")
    return (NUM_SAMPLES)

def encontraNumeroAluno(imgAluno):
    #encontra especificamente onde o número de aluno se encontra.
    imgAlunoGray = cv2.cvtColor(imgAluno, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(imgAlunoGray)

    #Aplica um close ao nr de aluno
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(notGray, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    max = cv2.contourArea(contours[0])
    b = contours[0]
    for c in contours:
        area = cv2.contourArea(c)
        if area > max:
            max = area
            b = c

    (x, y, w, h) = cv2.boundingRect(b)
    if (x != 0 and y != 0):
        roi = imgAluno[y: y + h, x: x + w]
    else:
        roi = imgAluno[y: y + h + 2, x: x + w + 3]

    return roi


def identificaNumeros(roi):
    #identifica um dado número
    roi = cv2.resize(roi, (60, 70))
    maior = 0
    num = 0
    for i in range(len(NUM_SAMPLES)):
        result = cv2.matchTemplate(roi, NUM_SAMPLES[i], cv2.TM_CCOEFF)
        (_, max_val, _, _) = cv2.minMaxLoc(result)
        if (max_val > maior):
            maior = max_val
            num = i

    return num


def confirmaAssinatura(assinatura):
    #Confirma se realmente existe uma assinatura, através do seu tamanho
    assinaturaGrey = cv2.cvtColor(assinatura, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(assinaturaGrey)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))

    closed = cv2.morphologyEx(notGray, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (10, 5))
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    maxW = 0
    coord = None
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > maxW:
            coord = (x, y, w, h)
            maxW = w

    return coord


def verificaAssinatura(assinatura):
    #Verifica se existe uma assiantura para um dado aluno primeiro pela percentagem de pixeis e depois, se necessário pelo tamanho da assinatura
    assinaturaGrey = cv2.cvtColor(assinatura, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(assinaturaGrey, 200, 255, cv2.THRESH_BINARY_INV)

    assinado = False
    incerto = False
    percentagemAssinado = (cv2.countNonZero(thresh) * 100) / thresh.size
    if percentagemAssinado > 10:
        assinado = True
    if percentagemAssinado <= 10 and percentagemAssinado >= 3:
        (x, y, w, h) = confirmaAssinatura(assinatura)
        if w > assinatura.shape[1] * 0.20 and h > assinatura.shape[0] * 0.4:
            assinado = True
        else:
            incerto = True

    return assinado, incerto


def extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas):
    #extrai linhas horizontais das tabelas para ser possível individualizar cada aluno
    (x, y, w, h) = roiAlunosLinhas
    roi = linhasHorizontais[y:y + h, x:x + w]
    linhasAlunos = []
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 30, minLineLength=400, maxLineGap=250)

    if len(lines) > 0:
        for line in lines:
            (x1, y1, x2, y2) = line[0]
            linhasAlunos.append((x1, y1, x2, y2))

        linhasAlunos = sorted(linhasAlunos, key=lambda y: y[1])

        linhasAlunosFinal = []
        (_, y, _, _) = linhasAlunos[0]
        for linha in linhasAlunos:
            (x1, y1, x2, y2) = linha
            if y1 - y > 10:
                linhasAlunosFinal.append(linha)
                y = y1
    else:
        quit("Folha ilegível, falha na divisão dos alunos")
    return linhasAlunosFinal


def processaAlunos(img):
    todosAlunos = []
    alunosPresentes = []
    imgFinal = img.copy()
    imgCerto = ff.carregaImagemCerto()

    linhasHorizontais, linhas = folhaPresenca.filtroDeLinhas(img)
    roiAlunos, roiAlunosLinhas = folhaPresenca.encontraTabelasAlunos(img, linhas)

    folhaInvalida = 0
    contador = 1;
    #percorre as tabelas dos alunos encontradas
    for r in range(len(roiAlunos)):
        linhasAlunos = extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas[r])
        larg = roiAlunos[r].shape[1]

        #percorre as linhas que definem cada espaço do aluno
        for i in range(len(linhasAlunos) - 1):
            (x1, Yi, x2, y2) = linhasAlunos[i]
            (x1, y1, x2, Yf) = linhasAlunos[i + 1]
            altura = Yf - Yi
            if altura < int(IMGY * 0.005):
                continue
            #extrai um aluno consuante as linhas
            Aluno = roiAlunos[r][Yi:Yf, 5:larg - 10]
            #extrai número de aluno consuante coordenadas fixas.
            nrAluno = Aluno[int(round(altura * 0.1)):int(round(altura * 0.93)),
                      int(round(larg * 0.1)):int(round(larg * 0.29))]
            #aproximação exata ao local do número de aluno
            nrAluno = encontraNumeroAluno(nrAluno)
            nrAlunoGray = cv2.cvtColor(nrAluno, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(nrAlunoGray, 130, 255, cv2.THRESH_BINARY_INV)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            if len(contours) > 0:
                contours = utils.sort_contours(contours, method="left-to-right")[0]
            larguraNumero = nrAluno.shape[1] / 10
            listaNr = []
            #percorre os números encontrados
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                roi = nrAlunoGray[y:y + h, x:x + w]
                roi = cv2.addWeighted(roi, 1, roi, 0, -35)
                if cv2.arcLength(c, True) > 25:
                    if (w > larguraNumero + (larguraNumero * 0.5)):
                        primeiroNumero = nrAlunoGray[y:y + h, x:int(x + (w / 2))]
                        primeiroNumero = cv2.addWeighted(primeiroNumero, 1, primeiroNumero, 0, -35)
                        segundoNumero = nrAlunoGray[y:y + h, int(x + (w / 2)):x + w]
                        segundoNumero = cv2.addWeighted(segundoNumero, 1, segundoNumero, 0, -35)
                        listaNr.append(identificaNumeros(primeiroNumero))
                        listaNr.append(identificaNumeros(segundoNumero))
                    else:
                        listaNr.append(identificaNumeros(roi))
            n = ""
            for j in listaNr:
                n = n + str(j)
            if len(n) == 10:
                try:
                    numero_Aluno = int(n)
                except ValueError:
                    print("Não foi possível ler o número do aluno corretamente")
                    numero_Aluno = 0

                assinatura = Aluno[int(round(altura * 0.25)):int(round(altura * 0.94)),
                             int(round(larg * 0.74)):int(round(larg * 0.97))]
                assinado, incerto = verificaAssinatura(assinatura)
                todosAlunos.append(numero_Aluno)
                if assinado:
                    ff.folhaFinal(imgFinal, roiAlunosLinhas[r], Yi, imgCerto)
                    alunosPresentes.append(numero_Aluno)

                print(
                    str(contador) + " - " + str(
                        numero_Aluno) + " = Assinatura incerta, verificar") if incerto else print(
                    str(contador) + " - " + str(numero_Aluno) + " = PRESENTE") if assinado else print(
                    str(contador) + " - " + str(numero_Aluno))
                contador += 1

            else:
                numero_Aluno = int(n)
                print(str(contador) + " - Não foi possível ler o número do aluno corretamente -> " + str(numero_Aluno))
                folhaInvalida += 1
                contador += 1
    if folhaInvalida > len(todosAlunos):
        quit("folha com problemas")

    imgFinal = cv2.resize(imgFinal, (1000, 1200))
    cv2.imshow("imagem Final", imgFinal)
    return todosAlunos, alunosPresentes

