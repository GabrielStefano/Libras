import os.path
import win32api
import cv2
import numpy as np
import pytesseract.pytesseract as ocr
from sympy import *
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
import re
import unidecode

path = 'F:\\Projeto Final\\Alfabeto Editado\\Alfabeto56.mp4'
setup = "alfabeto"  # setup de extração de legenda pode ser de "palavra", "alfabeto", "numerico", "frase" ou "naoembarcado"

if os.path.isfile(path) is False:
    print("Diretório não encontrado, verifique se o caminho do arquivo está correto.")
    os._exit(1)


def text_detection(image):
    orig = image.copy()
    (H, W) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    retangulos = 0
    areas = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        areas.append([startX, endX, startY, endY])
        retangulos = retangulos + 1

    # show the output image
    # cv2.imshow("Text Detection", orig)
    # cv2.waitKey(0)
    return areas, retangulos, orig


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [[x, y]]
        # refPt.append((x, y))
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append([x, y])
        cropping = False
        # draw a rectangle around the region of interest
        if refPt[0][1] != refPt[1][1] and refPt[0][0] != refPt[1][0]:
            cv2.rectangle(frame, (refPt[0][0], refPt[0][1]), (refPt[1][0], refPt[1][1]), (0, 255, 0), 2)
            cv2.imshow("image", frame)

        for i in range(retangulos):
            # if (refPt[0][1] + refPt[1][1]) / 2 > areas[i][2] and (refPt[0][1] + refPt[1][1]) / 2 < areas[i][3] and (refPt[0][0] + refPt[1][0]) / 2 > areas[i][0] and (refPt[0][0] + refPt[1][0]) / 2 < areas[i][1]:
            if (refPt[0][1]) > areas[i][2] and (refPt[1][1]) < areas[i][3] and (refPt[0][0]) > areas[i][0] and (
            refPt[1][0]) < areas[i][1]:
                cv2.rectangle(frame, (areas[i][0], areas[i][2]), (areas[i][1], areas[i][3]), (0, 255, 255), -1)
                refPt[0][0] = areas[i][0] + 2  # descontando as bordas retângulo que foi inserido (expessura 2)
                refPt[0][1] = areas[i][2] + 2
                refPt[1][0] = areas[i][1] - 2
                refPt[1][1] = areas[i][3] - 2


def save_frame(folder, legend, position):
    global countaux, dominant
    ver = 0
    caminho = None
    if os.path.isdir(folder + "\\" + legend) is True:
        if position > 0 and resultado[position - 1][1] == legend:
            caminho = folder + "\\" + legend
            # countaux = resultado[position-1][3]-resultado[position-1][2] + 1
            # print(resultado[position-1][3])
            # print(resultado[position-1][2])
            # print(countaux)
            ver = 1
        else:
            resultado[position] = "Posicao %d - " % (i + 1) + legend
            os.mkdir(folder + "\\Posicao %d - " % (i + 1) + legend)
            caminho = folder + "\\Posicao %d - " % (i + 1) + legend
            countaux = 1
    if os.path.isdir(folder + "\\" + legend) is False:
        os.mkdir(folder + "\\" + legend)
        caminho = folder + "\\" + legend
        countaux = 1

    for j in range(posicoes[position], posicoes[position + 1]):
        vidcap.set(1, j)
        ret, frame = vidcap.read()
        if setup != 'naoembarcado' and j == posicoes[position]:
            corte = frame[coord[0]:coord[1], coord[2]:coord[3]]
            pixels = np.float32(corte.reshape(-1, 3))
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            if ver == 0:
                dominant = palette[np.argmax(counts)]
            # print(dominant)
        # for w in range(coord[0], coord[1]+1):
        #     for x in range(coord[2], coord[3]+1):
        #         if (abs(average-(sum(frame[w][x])/3))>15):
        #             # print("to aqui")
        #             frame[w][x] = dominant
        # print(dominant)
        if setup != 'naoembarcado':
            cv2.rectangle(frame, (coord[2], coord[0]), (coord[3], coord[1]), (int(dominant[0]), int(dominant[1]), int(dominant[2])), -1)

        if countaux < 10:
            cv2.imwrite(caminho + "\\" + video + "-000%d.jpg" % countaux, frame)
        elif countaux > 9 and countaux < 100:
            cv2.imwrite(caminho + "\\" + video + "-00%d.jpg" % countaux, frame)
        else:
            cv2.imwrite(caminho + "\\" + video + "-0%d.jpg" % countaux, frame)
        # cv2.imwrite(caminho + "\\frame%d.jpg" % countaux, frame)
        countaux = countaux + 1


def exclusion_list(legend):
    lista = [' a ', ' as ', ' o ', ' os ', ' um ', ' uns ', ' uma ', ' umas ', ' que ', ' e ', ' em ']
    legend = ''.join(legend.splitlines())
    legend = legend.rjust(len(legend) + 1).ljust(len(legend) + 2)
    legend = legend.upper()
    for i in range(0, len(lista)):
        legend = legend.replace(lista[i].upper(), " ")
    return legend.strip()  # retorna a string sem espaçamento antes e depois


# --Inicializando vetores e parâmetros de operação
ys = []
xs = []
novo = []
posicoes = []
resultado = []
teste = []
count = 0
index = 1
# posicoes.append(0)
# success = True

vidcap = cv2.VideoCapture(path)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # quantidade total de frames do vídeo
fps = int(round(vidcap.get(cv2.CAP_PROP_FPS), 0))  # quantidade de frames por segundo
success, image0 = vidcap.read()
image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)  # converte a imagem para Escala de Cinza
image0 = cv2.threshold(image0, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

pasta = "imagens\\" + (path[len(path) - path[::-1].find('\\'):len(path):])  # crio uma pasta com o caminho do vídeo caso ela não exista
if os.path.isdir(pasta) is False:  # caso o diretório não exista ele é criado
    os.mkdir(pasta)
#
while count < (length - 2):
    # cv2.imwrite(pasta+"\\frame%d.jpg" % (count + 1), image0)
    success, image1 = vidcap.read()
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('a', image1)
    # cv2.waitKey(0)
    image1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow('b', image1)
    # cv2.waitKey(0)
    res = cv2.absdiff(image0, image1)  # calcula a diferença de pixel entre as imagens
    res = res.astype(np.uint8)  # convert the result to integer type ---
    percentage = (np.count_nonzero(res) * 100) / res.size  # find percentage difference based on number of pixels that are not zero ---
    ys.append(percentage)  # salva o vetor de percentuais
    count = count + 1
    image0 = image1

# cnn_benchmark = []
# for i in range(0,200):
#     cnn_benchmark.append(1)
novo = ys
a = path[len(path) - path[::-1].find('\\'):len(path):]
video = a[:len(a) - a[::-1].find('.') - 1:]
# plt.plot(ys[50:250])
# plt.plot(cnn_benchmark)
# plt.plot(43, ys[93], 'go')
# plt.plot(115, ys[165], 'go')
# plt.plot(191, ys[241], 'go')
# plt.plot(0, ys[50], 'ro')
# plt.plot(100, ys[150], 'ro')

#
# print(len(ys))
# plt.plot(ys)
# plt.plot(21, ys[21], 'go')
# plt.plot(71, ys[71], 'go')
# plt.plot(128, ys[128], 'go')
# plt.plot(168, ys[168], 'go')
# plt.plot(216, ys[216], 'go')
# plt.plot(279, ys[279], 'go')
# plt.plot(316, ys[316], 'go')
# plt.plot(365, ys[365], 'go')
# plt.plot(404, ys[404], 'go')
# plt.plot(441, ys[441], 'go')
# plt.plot(474, ys[474], 'go')
# plt.plot(509, ys[509], 'go')
# plt.plot(548, ys[548], 'go')
# plt.plot(597, ys[597], 'go')
# plt.plot(651, ys[651], 'go')
# plt.grid(True)
# plt.xlabel('${Frames}$')
# plt.ylabel('Variação (%)')
# plt.savefig("imagens\\grafico-" + video + ".png", format='png', dpi=1600)
# plt.show()

while (len(ys) - 1) % 4 != 0:  # interpolo um vetor de no máximo 4 graus, por isso preciso de um múltiplo de 4
    del (ys[len(ys) - 1])

xs = list(range(1, len(ys) + 1))
n = len(ys)
yi = 0
pol = 0
# i = 0
k = 0
c = 0
# max = (n - 1) / 4
set = 0
count = 0
x = symbols("x")

# while i < n:
#     termino = novo[i]
#     for j in range(k,(5+k)):
#         if i != j:
#             termino = termino*(x - xs[j])/(xs[i]-xs[j])
#     pol = termino + pol
#     if i % (4+k) == 0 and i != 0:
#         pol = expand(pol)
#         if k >= set:
#             if k > 0:
#                 del (teste[len(teste) - 1])
#             for l in range(k+1, (6+k)):
#                 teste.append(pol.subs(x, l))
#                 # if novo[l] >= 35 and novo[l-1] >= 25 and novo[l+1] >= 25 and diff(pol).subs(x,l)>0:
#                 # if l>1 and pol.subs(x,l) >= 10 and pol.subs(x,l-1) >= 10 and pol.subs(x,l+1) >= 10 and diff(pol).subs(x,l)>0:
#                 # if l>1 and pol.subs(x,l) >= 30 and pol.subs(x,l-1) >= 20 and pol.subs(x,l+1) >= 20 and diff(pol).subs(x,l)>0:
#                 # if pol.subs(x, l) >= 20 and diff(pol).subs(x, l) > 0:
#                 # if pol.subs(x, l) >= 0.6 and diff(pol).subs(x, l) > 0 and ys[l-1] > ys[l-2] and ys[l] < ys[l-1] and l < (n - 4):
#                 if pol.subs(x, l) >= 0.8 and diff(pol).subs(x, l) > 0 and l < (n - 4):
#                     somaantes = ys[l-2] + ys[l-3] + ys[l-4] + ys[l-5]
#                     somadepois = ys[l] + ys[l+1] + ys[l+2] + ys[l+3]
#                     if somadepois + somaantes + ys[l-1] >= 3.6:
#                         print(l, pol.subs(x, l))  # essas são as posições que identifiquei legenda
#                         posicoes.append(l)
#                         count = count + 1
#                         set = k + fps
#                         break
#         k = 4 + k
#         if i < (n-1):
#             i = i - 1
#             pol = 0
#     i = i+1

i = 0
# S/ LEGENDA = 0.5 E 2.5
# C/ LEGENDA = 1 E 4.5

while i < n:
    if ys[i] > 1 and i < (n - 4):
        somaantes = ys[i - 2] + ys[i - 3] + ys[i - 4] + ys[i - 5]
        somadepois = ys[i] + ys[i + 1] + ys[i + 2] + ys[i + 3]
        print(i)
        if somadepois + somaantes + ys[i - 1] >= 4.5:
            # print(i, ys[i])
            posicoes.append(i)
            i += int(fps)
            count += 1
    i += 1

posicoes.append(length)
count = count + index

print(posicoes)
print(count)
i = 0
j = 0
t = 0
if setup == "palavra":
    p = 0
    s = 0
    q = 0
    r = 0
    while i < count - 1:
        if i == 0 and t == 0:
            vidcap.set(1, (posicoes[i] + fps / 2))
            # vidcap.set(1, (posicoes[i]))
            ret, frame = vidcap.read()
            areas, retangulos, frame = text_detection(frame)

            position = posicoes[i] + fps / 2
            # global refPt, cropping
            win32api.MessageBox(0, 'Selecione uma área ou arraste o cursor para criar uma nova.'
                                   '\n c - continuar \n r - redesenhar \n n - ir para o próximo frame',
                                'Identifique a área da legenda')
            clone = frame.copy()
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback("image", click_and_crop)

            while True:
                # display the image and wait for a keypress
                cv2.imshow("image", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    frame = clone.copy()
                if key == ord("n"):
                    position += 1
                    vidcap.set(1, position)
                    ret, frame = vidcap.read()
                    clone = frame.copy()
                elif key == ord("c"):
                    break
                elif key == 27:
                    refPt = 0
                    break

            # if there are two reference points, then crop the region of interest
            # from teh image and display it
            if refPt != 0 and (refPt[0][1] > refPt[1][1]):
                aux = refPt[1][1]
                refPt[1][1] = refPt[0][1]
                refPt[0][1] = aux

            if refPt != 0 and (refPt[0][0] > refPt[1][0]):
                aux = refPt[1][0]
                refPt[1][0] = refPt[0][0]
                refPt[0][0] = aux

            if refPt != 0 and (refPt[0][1] == refPt[1][1] or refPt[0][0] == refPt[1][0]):
                print("Não é possível gerar um recorte a partir da área selecionada")
                # roi = clone

            coord = [refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]]

            cv2.destroyAllWindows()
        elif i == 0 and t != 0:
            vidcap.set(1, (posicoes[i] + fps / 2 + p))
            ret, frame = vidcap.read()
        elif i < count - 2:
            q = 0
            r = 0
            if posicoes[i + 1] - posicoes[i] > 2 * fps / 3:
                vidcap.set(1, (posicoes[i] + fps + p))
                ret, frame = vidcap.read()
                q = 1

            else:
                vidcap.set(1, (posicoes[i] + fps / 2 + p))
                ret, frame = vidcap.read()
                r = 1
        else:
            q = 0
            r = 0
            vidcap.set(1, ((posicoes[i] + posicoes[i + 1]) / 2 + p))
            ret, frame = vidcap.read()
            s = 1
        corte = frame[coord[0]:coord[1], coord[2]:coord[3]]
        cv2.imshow("frame", corte)
        cv2.waitKey(100)
        frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
        frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        legenda = ocr.image_to_string(frame, lang="por")
        # legenda = "ComLicenca"
        import re

        # if legenda != '' and len(legenda) > 1 and i < count and legenda.isalnum() is True:
        if legenda != '' and len(legenda) > 1 and i < count:
            legenda = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda)
            legenda = unidecode.unidecode(legenda)
            resultado.append(
                [path[len(path) - path[::-1].find('\\'):len(path):], legenda, posicoes[i], posicoes[i + 1]])
            print("Legenda - ", legenda.strip())
            save_frame(pasta, legenda.strip(), i)
            # save_frame(pasta, "Palavra %d - " % (i+1) + legenda.strip(), i)

        elif (legenda == '' or len(
                legenda) == 1 or legenda.isalnum() is False) and i < count:  # and i < count:  # aqui entram os tratamentos para determinar a legenda
            frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
            frame1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]
            legenda1 = ocr.image_to_string(frame1, lang="por")
            frame2 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)[1]
            legenda2 = ocr.image_to_string(frame2, lang="por")
            frame3 = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            legenda3 = ocr.image_to_string(frame3, lang="por")
            frame4 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO)[1]
            legenda4 = ocr.image_to_string(frame4, lang="por")

            if legenda1 != '' and len(legenda1) != 1:
                legenda1 = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda1)
                legenda1 = unidecode.unidecode(legenda1)
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda1, posicoes[i], posicoes[i + 1]])
                print("Legenda1 - ", legenda1.strip())
                save_frame(pasta, legenda1.strip(), i)

            elif legenda2 != '' and len(legenda2) != 1:
                legenda2 = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda2)
                legenda2 = unidecode.unidecode(legenda2)
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda2, posicoes[i], posicoes[i + 1]])
                print("Legenda2 - ", legenda2.strip())
                save_frame(pasta, legenda2.strip(), i)

            elif legenda3 != '' and len(legenda3) != 1:
                legenda3 = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda3)
                legenda3 = unidecode.unidecode(legenda3)
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda3, posicoes[i], posicoes[i + 1]])
                print("Legenda3 - ", legenda3.strip())
                save_frame(pasta, legenda3.strip(), i)

            elif legenda4 != '' and len(legenda4) != 1:
                # legenda4 = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda4)
                legenda4 = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda4)
                legenda4 = unidecode.unidecode(legenda4)
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda4, posicoes[i], posicoes[i + 1]])
                print("Legenda4 - ", legenda4.strip())
                save_frame(pasta, legenda4.strip(), i)

            else:
                # legenda5 = "Posicao %d" % (i+1)
                # resultado.append([path[len(path) - path[::-1].find('\\'):len(path):], legenda5, posicoes[i], posicoes[i + 1]])
                # save_frame(pasta, legenda5, i)

                if s == 1 and posicoes[i + 1] - ((posicoes[i] + posicoes[i + 1]) / 2 + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif q == 1 and posicoes[i + 1] - (posicoes[i] + fps + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif r == 1 and posicoes[i + 1] - (posicoes[i] + fps / 2 + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif (q == 1 and posicoes[i + 1] - (posicoes[i] + fps + p) <= fps / 3) or (
                        r == 1 and posicoes[i + 1] - (posicoes[i] + fps / 2 + p) < fps / 3) or (
                        s == 1 and posicoes[i + 1] - ((posicoes[i] + posicoes[i + 1]) / 2 + p) < fps / 3):
                    legenda5 = "Posicao %d" % (i + 1)
                    print(legenda5)
                    resultado.append(
                        [path[len(path) - path[::-1].find('\\'):len(path):], legenda5, posicoes[i], posicoes[i + 1]])
                    save_frame(pasta, legenda5, i)
                    p = 0
                else:
                    i = i - 1
                    t = 1
                    p = p + 1
        i = i + 1
    print(resultado)

elif setup == "alfabeto":
    p = 0
    s = 0
    q = 0
    r = 0
    while i < count - 1:
        if i == 0 and t == 0:
            vidcap.set(1, (posicoes[i] + fps / 2))
            ret, frame = vidcap.read()
            areas, retangulos, frame = text_detection(frame)

            position = posicoes[i] + fps / 2
            # global refPt, cropping
            win32api.MessageBox(0, 'Selecione uma área ou arraste o cursor para criar uma nova.'
                                   '\n c - continuar \n r - redesenhar \n n - ir para o próximo frame',
                                'Identifique a área da legenda')
            clone = frame.copy()
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback("image", click_and_crop)
            while True:
                # display the image and wait for a keypress
                cv2.imshow("image", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    frame = clone.copy()
                if key == ord("n"):
                    position += 1
                    vidcap.set(1, position)
                    ret, frame = vidcap.read()
                    clone = frame.copy()
                elif key == ord("c"):
                    break
                elif key == 27:
                    refPt = 0
                    break

            # if there are two reference points, then crop the region of interest
            # from teh image and display it
            if refPt != 0 and (refPt[0][1] > refPt[1][1]):
                aux = refPt[1][1]
                refPt[1][1] = refPt[0][1]
                refPt[0][1] = aux

            if refPt != 0 and (refPt[0][0] > refPt[1][0]):
                aux = refPt[1][0]
                refPt[1][0] = refPt[0][0]
                refPt[0][0] = aux

            if refPt != 0 and (refPt[0][1] == refPt[1][1] or refPt[0][0] == refPt[1][0]):
                print("Não é possível gerar um recorte a partir da área selecionada")
                # roi = clone

            coord = [refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]]

            cv2.destroyAllWindows()
        elif i == 0 and t != 0:
            vidcap.set(1, (posicoes[i] + fps / 2 + p))
            ret, frame = vidcap.read()
        elif i < count - 2:
            q = 0
            r = 0
            if posicoes[i + 1] - posicoes[i] > 2 * fps / 3:
                vidcap.set(1, (posicoes[i] + fps + p))
                ret, frame = vidcap.read()
                q = 1

            else:
                vidcap.set(1, (posicoes[i] + fps / 2 + p))
                ret, frame = vidcap.read()
                r = 1
        else:
            q = 0
            r = 0
            vidcap.set(1, ((posicoes[i] + posicoes[i + 1]) / 2 + p))
            ret, frame = vidcap.read()
            s = 1

        corte = frame[coord[0]:coord[1], coord[2]:coord[3]]
        cv2.imshow("frame", corte)
        cv2.waitKey(100)
        frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
        frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        legenda = ocr.image_to_string(frame, lang="por", config='--psm 10')
        legenda = re.sub(u'[^a-zA-Z ]', '', legenda.upper())
        if legenda != '' and len(legenda) == 1 and i < count and legenda.isalnum() is True:
            resultado.append(
                [path[len(path) - path[::-1].find('\\'):len(path):], legenda, posicoes[i], posicoes[i + 1]])
            print("Legenda - ", legenda.strip())
            save_frame(pasta, legenda.strip(), i)
            p = 0
        elif (legenda == '' or len(
                legenda) > 1 or legenda.isalnum() is False) and i < count:  # and i < count:  # aqui entram os tratamentos para determinar a legenda
            frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)

            frame1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]
            legenda1 = ocr.image_to_string(frame1, config='--psm 10')
            legenda1 = re.sub(u'[^a-zA-Z ]', '', legenda1.upper())
            frame2 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)[1]
            legenda2 = ocr.image_to_string(frame2, config='--psm 10')
            legenda2 = re.sub(u'[^a-zA-Z ]', '', legenda2.upper())
            frame3 = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            legenda3 = ocr.image_to_string(frame3, config='--psm 10')
            legenda3 = re.sub(u'[^a-zA-Z ]', '', legenda3.upper())
            frame4 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO)[1]
            legenda4 = ocr.image_to_string(frame4, config='--psm 10')
            legenda4 = re.sub(u'[^a-zA-Z ]', '', legenda4.upper())

            if legenda1 != '' and len(legenda1) == 1 and legenda1.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda1, posicoes[i], posicoes[i + 1]])
                print("Legenda 1 - ", legenda1.strip())
                save_frame(pasta, legenda1.strip(), i)
                p = 0
            elif legenda2 != '' and len(legenda2) == 1 and legenda2.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda2, posicoes[i], posicoes[i + 1]])
                print("Legenda 2 - ", legenda2.strip())
                save_frame(pasta, legenda2.strip(), i)
                p = 0
            elif legenda3 != '' and len(legenda3) == 1 and legenda3.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda3, posicoes[i], posicoes[i + 1]])
                print("Legenda 3 - ", legenda3.strip())
                save_frame(pasta, legenda3.strip(), i)
                p = 0
            elif legenda4 != '' and len(legenda4) == 1 and legenda4.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda4, posicoes[i], posicoes[i + 1]])
                print("Legenda 4 - ", legenda4.strip())
                save_frame(pasta, legenda4.strip(), i)
                p = 0
            else:
                # legenda5 = "Posicao %d" % (i+1)
                # resultado.append([path[len(path) - path[::-1].find('\\'):len(path):], legenda5, posicoes[i], posicoes[i + 1]])
                # save_frame(pasta, legenda5, i)

                if s == 1 and posicoes[i + 1] - ((posicoes[i] + posicoes[i + 1]) / 2 + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif q == 1 and posicoes[i + 1] - (posicoes[i] + fps + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif r == 1 and posicoes[i + 1] - (posicoes[i] + fps / 2 + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif (q == 1 and posicoes[i + 1] - (posicoes[i] + fps + p) <= fps / 3) or (
                        r == 1 and posicoes[i + 1] - (posicoes[i] + fps / 2 + p) < fps / 3) or (
                        s == 1 and posicoes[i + 1] - ((posicoes[i] + posicoes[i + 1]) / 2 + p) < fps / 3):
                    legenda5 = "Posicao %d" % (i + 1)
                    print(legenda5)
                    resultado.append(
                        [path[len(path) - path[::-1].find('\\'):len(path):], legenda5, posicoes[i], posicoes[i + 1]])
                    save_frame(pasta, legenda5, i)
                    p = 0
                else:
                    i = i - 1
                    t = 1
                    p = p + 1
        i = count
    print(resultado)

elif setup == "numerico":
    p = 0
    s = 0
    q = 0
    r = 0
    while i < count - 1:
        if i == 0 and t == 0:
            vidcap.set(1, (posicoes[i] + fps / 2))
            ret, frame = vidcap.read()
            areas, retangulos, frame = text_detection(frame)

            position = posicoes[i] + fps / 2
            # global refPt, cropping
            win32api.MessageBox(0, 'Selecione uma área ou arraste o cursor para criar uma nova.'
                                   '\n c - continuar \n r - redesenhar \n n - ir para o próximo frame',
                                'Identifique a área da legenda')
            clone = frame.copy()
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback("image", click_and_crop)
            while True:
                # display the image and wait for a keypress
                cv2.imshow("image", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    frame = clone.copy()
                if key == ord("n"):
                    position += 1
                    vidcap.set(1, position)
                    ret, frame = vidcap.read()
                    clone = frame.copy()
                elif key == ord("c"):
                    break
                elif key == 27:
                    refPt = 0
                    break

            # if there are two reference points, then crop the region of interest
            # from teh image and display it
            if refPt != 0 and (refPt[0][1] > refPt[1][1]):
                aux = refPt[1][1]
                refPt[1][1] = refPt[0][1]
                refPt[0][1] = aux

            if refPt != 0 and (refPt[0][0] > refPt[1][0]):
                aux = refPt[1][0]
                refPt[1][0] = refPt[0][0]
                refPt[0][0] = aux

            if refPt != 0 and (refPt[0][1] == refPt[1][1] or refPt[0][0] == refPt[1][0]):
                print("Não é possível gerar um recorte a partir da área selecionada")
                # roi = clone

            coord = [refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]]

            cv2.destroyAllWindows()
        elif i == 0 and t != 0:
            vidcap.set(1, (posicoes[i] + fps / 2 + p))
            ret, frame = vidcap.read()
        elif i < count - 2:
            q = 0
            r = 0
            if posicoes[i + 1] - posicoes[i] > 2 * fps / 3:
                vidcap.set(1, (posicoes[i] + fps + p))
                ret, frame = vidcap.read()
                q = 1

            else:
                vidcap.set(1, (posicoes[i] + fps / 2 + p))
                ret, frame = vidcap.read()
                r = 1
        else:
            q = 0
            r = 0
            vidcap.set(1, ((posicoes[i] + posicoes[i + 1]) / 2 + p))
            ret, frame = vidcap.read()
            s = 1

        corte = frame[coord[0]:coord[1], coord[2]:coord[3]]
        cv2.imshow("frame", corte)
        cv2.waitKey(100)
        frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
        frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        legenda = ocr.image_to_string(frame, config='--psm 10')
        legenda = re.sub(u'[^0123456789 ]', '', legenda.upper())

        if legenda != '' and i < count and legenda.isalnum() is True:
            resultado.append(
                [path[len(path) - path[::-1].find('\\'):len(path):], legenda, posicoes[i], posicoes[i + 1]])
            print("Legenda - ", legenda.strip())
            save_frame(pasta, legenda.strip(), i)
            p = 0
        elif (
                legenda == '' or legenda.isalnum() is False) and i < count:  # and i < count:  # aqui entram os tratamentos para determinar a legenda
            frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)

            frame1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]
            legenda1 = ocr.image_to_string(frame1, config='--psm 10')
            legenda1 = re.sub(u'[^0123456789 ]', '', legenda1.upper())
            frame2 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)[1]
            legenda2 = ocr.image_to_string(frame2, config='--psm 10')
            legenda2 = re.sub(u'[^0123456789 ]', '', legenda2.upper())
            frame3 = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            legenda3 = ocr.image_to_string(frame3, config='--psm 10')
            legenda3 = re.sub(u'[^0123456789 ]', '', legenda3.upper())
            frame4 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO)[1]
            legenda4 = ocr.image_to_string(frame4, config='--psm 10')
            legenda4 = re.sub(u'[^0123456789 ]', '', legenda4.upper())

            if legenda1 != '' and legenda1.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda1, posicoes[i], posicoes[i + 1]])
                print("Legenda 1 - ", legenda1.strip())
                save_frame(pasta, legenda1.strip(), i)
                p = 0
            elif legenda2 != '' and legenda2.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda2, posicoes[i], posicoes[i + 1]])
                print("Legenda 2 - ", legenda2.strip())
                save_frame(pasta, legenda2.strip(), i)
                p = 0
            elif legenda3 != '' and legenda3.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda3, posicoes[i], posicoes[i + 1]])
                print("Legenda 3 - ", legenda3.strip())
                save_frame(pasta, legenda3.strip(), i)
                p = 0
            elif legenda4 != '' and legenda4.isalnum() is True:
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda4, posicoes[i], posicoes[i + 1]])
                print("Legenda 4 - ", legenda4.strip())
                save_frame(pasta, legenda4.strip(), i)
                p = 0
            else:
                if s == 1 and posicoes[i + 1] - ((posicoes[i] + posicoes[i + 1]) / 2 + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif q == 1 and posicoes[i + 1] - (posicoes[i] + fps + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif r == 1 and posicoes[i + 1] - (posicoes[i] + fps / 2 + p) > fps / 3:
                    p = p + 1
                    i = i - 1

                elif (q == 1 and posicoes[i + 1] - (posicoes[i] + fps + p) <= fps / 3) or (
                        r == 1 and posicoes[i + 1] - (posicoes[i] + fps / 2 + p) < fps / 3) or (
                        s == 1 and posicoes[i + 1] - ((posicoes[i] + posicoes[i + 1]) / 2 + p) < fps / 3):
                    legenda5 = "Posicao %d" % (i + 1)
                    print(legenda5)
                    resultado.append(
                        [path[len(path) - path[::-1].find('\\'):len(path):], legenda5, posicoes[i], posicoes[i + 1]])
                    save_frame(pasta, legenda5, i)
                    p = 0
                else:
                    i = i - 1
                    t = 1
                    p = p + 1
        i = i + 1
    print(resultado)

elif setup == "frase":
    p = 0
    s = 0
    q = 0
    r = 0
    while i < count - 1:
        if i == 0 and t == 0:
            vidcap.set(1, (posicoes[i] + fps / 2))
            ret, frame = vidcap.read()
            areas, retangulos, frame = text_detection(frame)

            position = posicoes[i] + fps / 2
            # global refPt, cropping
            win32api.MessageBox(0, 'Selecione uma área ou arraste o cursor para criar uma nova.'
                                   '\n c - continuar \n r - redesenhar \n n - ir para o próximo frame',
                                'Identifique a área da legenda')
            clone = frame.copy()
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback("image", click_and_crop)
            while True:
                # display the image and wait for a keypress
                cv2.imshow("image", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    frame = clone.copy()
                if key == ord("n"):
                    position += 1
                    vidcap.set(1, position)
                    ret, frame = vidcap.read()
                    clone = frame.copy()
                elif key == ord("c"):
                    break
                elif key == 27:
                    refPt = 0
                    break

            # if there are two reference points, then crop the region of interest
            # from teh image and display it
            if refPt != 0 and (refPt[0][1] > refPt[1][1]):
                aux = refPt[1][1]
                refPt[1][1] = refPt[0][1]
                refPt[0][1] = aux

            if refPt != 0 and (refPt[0][0] > refPt[1][0]):
                aux = refPt[1][0]
                refPt[1][0] = refPt[0][0]
                refPt[0][0] = aux

            if refPt != 0 and (refPt[0][1] == refPt[1][1] or refPt[0][0] == refPt[1][0]):
                print("Não é possível gerar um recorte a partir da área selecionada")
                # roi = clone

            coord = [refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]]

            cv2.destroyAllWindows()
        elif i == 0 and t != 0:
            vidcap.set(1, (posicoes[i] + fps / 2 + p))
            ret, frame = vidcap.read()
        elif i < count - 2:
            q = 0
            r = 0
            if posicoes[i + 1] - posicoes[i] > 2 * fps / 3:
                vidcap.set(1, (posicoes[i] + fps + p))
                ret, frame = vidcap.read()
                q = 1

            else:
                vidcap.set(1, (posicoes[i] + fps / 2 + p))
                ret, frame = vidcap.read()
                r = 1
        else:
            q = 0
            r = 0
            vidcap.set(1, ((posicoes[i] + posicoes[i + 1]) / 2 + p))
            ret, frame = vidcap.read()
            s = 1

        corte = frame[coord[0]:coord[1], coord[2]:coord[3]]
        cv2.imshow("frame", corte)
        cv2.waitKey(100)
        if i == 0:
            frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
            frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            legenda = ocr.image_to_string(frame, lang="eng")
            legenda = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda.upper())
            legenda = exclusion_list(legenda)
            legenda = unidecode.unidecode(legenda)
            print(legenda)
            if legenda != '' and len(legenda) > 1 and i < count:
                if (len(legenda.split())) == count - 1:
                    print("It's a match")
                    legenda = legenda.split()
                    auxcount = 0
                    for k in range(i, count - 1):
                        resultado.append(
                            [path[len(path) - path[::-1].find('\\'):len(path):], legenda[auxcount], posicoes[k],
                             posicoes[k + 1]])
                        save_frame(pasta, legenda[auxcount], k)
                        auxcount = auxcount + 1
                    i = count
                elif (len(legenda.split())) > count - 1:
                    auxcount = 0
                    for k in range(i, count - 1):
                        resultado.append(
                            [path[len(path) - path[::-1].find('\\'):len(path):], legenda[auxcount], posicoes[k],
                             posicoes[k + 1]])
                        save_frame(pasta, legenda + " - %d" % k, k)
                        auxcount = auxcount + 1
                    i = count

                elif (len(legenda.split())) < count - 1:
                    legenda = legenda.split()
                    auxcount = 0
                    for k in range(i, len(legenda)):
                        resultado.append(
                            [path[len(path) - path[::-1].find('\\'):len(path):], legenda[auxcount], posicoes[k],
                             posicoes[k + 1]])
                        save_frame(pasta, legenda[auxcount], k)
                        auxcount = auxcount + 1
                    for k in range(len(legenda), count-1):
                        legenda = "Posicao %d" % (k + 1)
                        resultado.append(
                            [path[len(path) - path[::-1].find('\\'):len(path):], legenda, posicoes[k], posicoes[k + 1]])
                        save_frame(pasta, legenda, k)
                    i = count
            else:
                if posicoes[i] + fps / 2 + p < posicoes[i + 1] - 5:
                    for k in range(i, count - 1):
                        resultado.append(
                            [path[len(path) - path[::-1].find('\\'):len(path):], legenda, posicoes[k], posicoes[k + 1]])
                        save_frame(pasta, "Legenda Indeterminada - %d" % (k + 1), k)
                    i = count
                else:
                    t = 1
                    p = p + 1
                    i = i - 1
        #
        # contador_sinais = 1
        # if legenda != '' and len(legenda) > 1 and i < count and legenda.isalnum() is True:
        #     legenda_ini = exclusion_list(legenda)
        #     contador_sinais = 0
        #     legenda_fim = legenda_ini
        #     k = i
        # elif (legenda == '' or len(legenda) == 1 or legenda.isalnum() is False) and i < count:
        #     i = i - 1
        #     t = 1
        #     p = p + 1
        #
        # if contador_sinais == 0:
        #     while legenda_ini == legenda_fim: # enquanto a legenda permanecer igual
        #         vidcap.set(1, (posicoes[k+1] + fps/2))
        #         ret, frame = vidcap.read()
        #         corte = frame[coord[0]:coord[1], coord[2]:coord[3]]
        #         frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
        #         frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #         legenda = ocr.image_to_string(frame, lang="por")
        #         legenda = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda.upper())
        #         legenda_fim = exclusion_list(legenda)
        #         contador_sinais = contador_sinais + 1
        #         k = k + 1
        #
        # if contador_sinais == (len(legenda.split())):
        #     print("It's a match")
        #     legenda = legenda_ini.split()
        #     auxcount = 0
        #     for k in range(i, i + (legenda.ini(' ')+1)):
        #         resultado.append([path[len(path) - path[::-1].find('\\'):len(path):], legenda[auxcount], posicoes[k], posicoes[k + 1]])
        #         save_frame(pasta, legenda[auxcount], k)
        #         auxcount = auxcount + 1
        #     i = k
        # else:
        #     print("Revise sua lista de exclusão")
        # i = i + 1

elif setup == "naoembarcado":
    p = 0
    s = 0
    q = 0
    r = 0
    # while i < count - 1:
    # if i == 0 and t == 0:
    #     vidcap.set(1, (posicoes[i] + fps/2))
    #     ret, frame = vidcap.read()
    #     areas, retangulos, frame = text_detection(frame)
    #
    #     position = posicoes[i] + fps/2
    #     # global refPt, cropping
    #     win32api.MessageBox(0, 'Selecione uma área ou arraste o cursor para criar uma nova.'
    #                            '\n c - continuar \n r - redesenhar \n n - ir para o próximo frame',
    #                         'Identifique a área da legenda')
    #     clone = frame.copy()
    #     cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    #     cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #     cv2.setMouseCallback("image", click_and_crop)
    #     while True:
    #         # display the image and wait for a keypress
    #         cv2.imshow("image", frame)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("r"):
    #             frame = clone.copy()
    #         if key == ord("n"):
    #             position += 1
    #             vidcap.set(1, position)
    #             ret, frame = vidcap.read()
    #             clone = frame.copy()
    #         elif key == ord("c"):
    #             break
    #         elif key == 27:
    #             refPt = 0
    #             break
    #
    #     # if there are two reference points, then crop the region of interest
    #     # from teh image and display it
    #     if refPt != 0 and (refPt[0][1] > refPt[1][1]):
    #         aux = refPt[1][1]
    #         refPt[1][1] = refPt[0][1]
    #         refPt[0][1] = aux
    #
    #     if refPt != 0 and (refPt[0][0] > refPt[1][0]):
    #         aux = refPt[1][0]
    #         refPt[1][0] = refPt[0][0]
    #         refPt[0][0] = aux
    #
    #     if refPt != 0 and (refPt[0][1] == refPt[1][1] or refPt[0][0] == refPt[1][0]):
    #         print("Não é possível gerar um recorte a partir da área selecionada")
    #         # roi = clone
    #
    #     coord = [refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]]
    #
    #     cv2.destroyAllWindows()
    # elif i == 0 and t != 0:
    #     vidcap.set(1, (posicoes[i] + fps/2 + p))
    #     ret, frame = vidcap.read()
    # elif i < count-2:
    #     q = 0
    #     r = 0
    #     if posicoes[i+1] - posicoes[i] > 2*fps/3:
    #         vidcap.set(1, (posicoes[i] + fps + p))
    #         ret, frame = vidcap.read()
    #         q = 1
    #
    #     else:
    #         vidcap.set(1, (posicoes[i] + fps/2 + p))
    #         ret, frame = vidcap.read()
    #         r = 1
    # else:
    #     q = 0
    #     r = 0
    #     vidcap.set(1, ((posicoes[i] + posicoes[i+1])/2 + p))
    #     ret, frame = vidcap.read()
    #     s = 1
    #
    # corte = frame[coord[0]:coord[1], coord[2]:coord[3]]
    # cv2.imshow("frame", corte)
    # cv2.waitKey(100)

    # frame = cv2.cvtColor(corte, cv2.COLOR_BGR2GRAY)
    # frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # legenda = ocr.image_to_string(frame, lang="por")
    # legenda = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', '', legenda.upper())

    local = (path[:len(path) - path[::-1].find('\\'):])
    with open(local + video + '.srt', 'r') as legenda:
        linhas = sum(1 for line in legenda)
    legenda = open(local + video + '.srt', encoding='utf-8', mode='r')
    tempos = []
    legendas = []
    for i in range(0, linhas):
        linha = legenda.readline()
        if (i + 3) % 4 == 0:
            tempos.append(linha)
        if (i + 2) % 4 == 0:
            legendas.append(linha)

    j = 0
    print(legendas)
    print(tempos)
    for i in range(0, len(legendas)):
        legenda = exclusion_list(legendas[i])
        legenda = unidecode.unidecode(legenda)

        tempo1 = tempos[i]
        tempo2 = tempos[i + 1]
        d1 = (tempo1[:tempo1[::].find(' '):])
        h1 = (d1[:d1[::].find(':'):])
        m1 = (tempo1[3:5:])
        s1 = (d1[d1[::-1].find(':')::])
        s1 = s1.replace(',', '.')
        s1 = round(float(s1), 1)
        d1 = int(h1)*60*60*fps + int(m1)*60*fps + s1*fps

        d2 = (tempo2[:tempo2[::].find(' '):])
        h2 = (d2[:d2[::].find(':'):])
        m2 = (tempo2[3:5:])
        s2 = (d2[d2[::-1].find(':')::])
        s2 = s2.replace(',', '.')
        s2 = round(float(s2), 1)
        d2 = int(h2)*60*60*fps + int(m2)*60*fps + s2*fps
        print(legenda)
        d1 = int(d1)
        d2 = int(d2)
        print(d1, d2)
        k = 1
        while j < len(posicoes):
            if posicoes[j] >= d1 and posicoes[j + 1] <= d2:
                legenda1 = "Sinal %d - " % k + legenda
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda1, posicoes[j], posicoes[j + 1]])
                save_frame(pasta, legenda1.strip(), j)
                j = j + 1
            elif posicoes[j] >= d1 and posicoes[j + 1] > d2:
                legenda1 = "Sinal %d - " % k + legenda
                resultado.append(
                    [path[len(path) - path[::-1].find('\\'):len(path):], legenda1, posicoes[j], d2])
                # posicoes[j + 1] = int(d2 * fps)
                posicoes.insert(j + 1, d2)
                save_frame(pasta, legenda1.strip(), j)
                j = j + 1
                break
            elif posicoes[j] <= d1:
                break
            k = k + 1