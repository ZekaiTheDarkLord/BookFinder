from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def getWordsFromImage(img_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
    # img_path = 'library_pictures/masked4.png'
    result = ocr.ocr(img_path, cls=True)
    # print(result)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    result = result[0]
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')

    txts = [line[1][0] for line in result]

    return txts


getWordsFromImage('library_pictures/masked4.png')
