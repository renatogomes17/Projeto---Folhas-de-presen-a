import sys
import cv2
import utils
import numpy as np
import imutils

IMGX = utils.IMGX
IMGY = utils.IMGY

def carregaImagem():
    #carrega a iamgem da folha de presenças com o formato RGB
    if len(sys.argv) > 1:
        img = cv2.imread(str(sys.argv[1]))
        if img is not None:
            #aplica um redimensionamento à imagem.
            img = cv2.resize(img, (utils.IMGX, utils.IMGY))
        else:
            quit("erro ao carregar folha de presença")
    else:
        quit("erro ao carregar folha de presença")
    return img

def corrigeAlinhamento(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY_INV)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = utils.sort_contours(contours, method="top-to-bottom")[0]

    if len(contours) > 0:
        todosContornos = []
        for c in contours:
            perimetro = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            if (perimetro > 1000 and area > 100000):
                (x, y, w, h) = cv2.boundingRect(c)
                roi = img[y:y + h, x:x + w]
                todosContornos.append(roi)

        cabecalho = todosContornos[0]

        grayCabecalho = cv2.cvtColor(cabecalho, cv2.COLOR_BGR2GRAY)
        grayCabecalho = np.float32(grayCabecalho)
        corners = cv2.goodFeaturesToTrack(grayCabecalho, 500, 0.02, 20)
        corners = np.int0(corners)
        if len(corners) > 0:
            leftCorners = []
            rightCorners = []
            larguraCabecalho = cabecalho.shape[1]

            for corner in corners:
                x, y = corner.ravel()
                if x < 50:
                    leftCorners.append(y)
                if x > (larguraCabecalho - 50):
                    rightCorners.append(y)
            leftCorners.sort()
            rightCorners.sort()
            leftCorner = leftCorners[0]
            rightCorner = rightCorners[0]
            correcao = 0
            if leftCorner > rightCorner:
                correcao = -leftCorner
            if rightCorner > leftCorner:
                correcao = rightCorner

            correcao = correcao * 0.033
            print("correcao ->" + str(correcao))
            img = imutils.rotate(img, correcao)
            img = img[int(IMGY * 0.05):IMGY - int(IMGY * 0.05), int(IMGX * 0.02):IMGX - int(IMGX * 0.02)]

    else:
        quit("Folha ilegível, falha no alinhamento")
    return img

def filtroDeLinhas(img):
    #passa a imagem para uma escala de cinzento e de seguida converte para o inverso.
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(imgray)
    #aplica um threshold à imagem
    adpThresh = cv2.adaptiveThreshold(notGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -6)

    horizontal = np.copy(adpThresh)
    vertical = np.copy(adpThresh)
    #cria um kernell rectangular e de seguida aplica a op. morfológica OPEN à imagem (OPEN = erosão seuida de uma dilatação) para retirar as linhas horizontais
    kernelHorizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, kernelHorizontal)

    # repete o processo, mas desta vez com um kernell que permite retirar apena as linhas verticais
    kernelVeritcal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, kernelVeritcal)

    #junta as duas imagens numa só
    VerticalHorizontal = cv2.addWeighted(vertical, 1, horizontal, 1, 0)

    return horizontal, VerticalHorizontal

def encontraTabelasAlunos(img, linhas):
    #Procura todos os contornos externos da imagem com as linhas
    contours = cv2.findContours(linhas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = utils.sort_contours(contours, method="top-to-bottom")[0]

    if len(contours) > 0:
        todosContornos = []
        todosContornosLinhas = []
        roiAlunos = []
        roiAlunosLinhas = []
        for c in contours:
            perimetro = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            if (perimetro > 1000 and area > 50000):
                (x, y, w, h) = cv2.boundingRect(c)
                roi = img[y:y + h, x:x + w]
                todosContornos.append(roi)
                todosContornosLinhas.append((x, y, w, h))

        if len(todosContornos) > 0:
            todosContornosLinhas.pop(0)
            todosContornos.pop(0)
            roiAlunosLinhas = todosContornosLinhas
            roiAlunos = todosContornos

            if len(roiAlunos) > 1:
                (x1,_,_,_) = todosContornosLinhas[0]
                (x2, _, _, _) = todosContornosLinhas[1]
                if (x1 > x2):
                    temp = roiAlunos[0]
                    roiAlunos[0] = roiAlunos[1]
                    roiAlunos[1] = temp
                    temp = roiAlunosLinhas[0]
                    roiAlunosLinhas[0] = roiAlunosLinhas[1]
                    roiAlunosLinhas[1] = temp

        if (roiAlunos[0].shape[1] > IMGX * 0.65):
            largura = roiAlunos[0].shape[1];
            altura = roiAlunos[0].shape[0]
            roi1 = roiAlunos[0][0:altura, 0:int(largura / 2)]
            roi2 = roiAlunos[0][0:altura, int(largura / 2):largura]
            roiAlunos = []
            roiAlunos.append(roi1)
            roiAlunos.append(roi2)

            (x, y, w, h) = roiAlunosLinhas[0]
            roiLinhas1 = (x, y, int(w / 2), h)
            roiLinhas2 = (int(w / 2), y, w, h)
            roiAlunosLinhas = []
            roiAlunosLinhas.append(roiLinhas1)
            roiAlunosLinhas.append(roiLinhas2)
    else:
        quit("Folha ilegível, não foi possivel encontrar nenhuma tabela de alunos")

    return roiAlunos, roiAlunosLinhas