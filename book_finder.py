import cv2
import numpy as np
from matplotlib import pyplot as plt

import get_box
import get_words
import imageseg_sam
from rapidfuzz import process, fuzz


def get_partial_score(expect, word):
    return fuzz.ratio(expect, word)


def search_highest_matching_string(choices, word):
    process.extractOne(word, choices)


def show_book(score_to_box, scores, image):
    # get the largest score
    scores = np.array(scores)
    largest_score = scores.max()
    best_fit_bounding_box = score_to_box[largest_score]

    plt.imshow(image)
    imageseg_sam.show_box(best_fit_bounding_box, plt.gca())
    plt.axis('off')

    plt.show()


def find_book_for_box(img_path, words, current_box=[0, 0, 0, 0]):
    score_to_box = {}
    scores = []
    cropped_image = cv2.imread(img_path)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    # get the bounding box for each single book inside the cropped image
    partial_bounding_box = np.array(get_box.get_bounding_box_for_single_book(img_path))

    # get a list of masks for single books
    image_changed, masks = imageseg_sam.get_mask(img_path, partial_bounding_box)

    partial_box_index = 0

    # iterate each mask
    for mask in masks:
        # generate a image with only the book
        selectedBook = imageseg_sam.get_selected(cropped_image, mask)
        cv2.imwrite('book.jpg', selectedBook)

        # get the text in this image
        text = get_words.getWordsFromImage('book.jpg')
        text = ''.join(text)

        # get the score of this image then add to list and hashmap
        score = get_partial_score(words, text)
        scores.append(score)

        # get the box in the whole image
        box_at_image = partial_bounding_box[partial_box_index]
        box_at_image = [box_at_image[0] + current_box[0], box_at_image[1] + current_box[1],
                        box_at_image[2] + current_box[0], box_at_image[3] + current_box[1]]
        score_to_box[score] = box_at_image

    return score_to_box, scores


def find_book_in_bookshelf(img_path, words):
    score_to_box = {}
    scores = []
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get bounding box of bookshelf
    bounding_box = np.array(get_box.get_bounding_box_for_bookshell(img_path))

    # iterate each box
    for current_box in bounding_box:
        # crop the image by the box and write to local
        cropped_image = image[current_box[1]:current_box[3], current_box[0]:current_box[2]]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('output.jpg', cropped_image)
        partial_score_to_box, partial_scores = find_book_for_box('output.jpg', words, current_box)
        score_to_box.update(partial_score_to_box)
        scores.append(partial_scores)

    return score_to_box, scores
