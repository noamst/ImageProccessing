import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import os
from PIL import Image, ImageOps
from torchvision import transforms
from torch import nn
from collections import OrderedDict
from simple_colors import *

global_rectangles = []
selected_indices = []
rect_to_word = {}
rect_to_position = {}

current_status = {
    'line': 0,
    'position': 0
}


class EMNISTCNN(nn.Module):
    def __init__(self, fmaps1, fmaps2, dense, dropout):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=fmaps1, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=fmaps1, out_channels=fmaps2, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fcon1 = nn.Sequential(nn.Linear(7 * 7 * fmaps2, dense), nn.LeakyReLU())
        self.fcon2 = nn.Linear(dense, 27)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x


model = EMNISTCNN(fmaps1=40, fmaps2=160, dense=200, dropout=0.5)

checkpoint = torch.load('models/torch_emnistcnn_latest.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def gui():
    dict2 = {}
    large_img = None
    global rect_to_position
    while True:
        try:
            print(green("\nPlease choose an option:"))
            print("1) Load image")
            print("2) Annotate by clicking the image")
            print("3) Annotate by typing the words")
            print("4) Exit")
            option = int(input("\nYour choice: "))
            # other code
        except ValueError:
            print(red("Invalid input. Please enter an integer.\n"))
            continue

        if option == 1:
            if option == 1:
                contours, large_img = openImageFindContours()

                if contours is None or large_img is None:
                    continue
                letter_rects = find_letters_rect(contours)
                word_rects = find_word_rects(letter_rects)
                rect_to_position = map_rectangles_to_lines_and_positions(word_rects)

                letter_to_word = map_rects_to_words(word_rects, letter_rects)
                dict = crop_letters_from_image(contours, large_img)
                dict2 = generate_words(letter_to_word, dict)
                dict2 = OrderedDict(reversed(list(dict2.items())))

        elif option == 2:
            if large_img is not None and dict2 is not None:
                cv2.namedWindow("image")
                cv2.setMouseCallback("image", click_and_crop, param=[large_img, dict2])
                cv2.imshow("image", large_img)
                print("Press any key to continue...")
                cv2.waitKey(0)
            else:
                print(red("\nYou need to load an image first."))
        elif option == 3:
            if large_img is not None and dict2 is not None:
                print(green("Enter your text:"))
                text = input()
                words = text.split()

                for word in words:
                    rect = find_matching_rect(dict2, word)
                    if rect is not None:
                        x_min, y_min, x_max, y_max = rect
                        cv2.rectangle(large_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        if rect in dict2:
                            del dict2[rect]
                    else:
                        print(red(f"\nNo rectangle found for word: {word}"))

                cv2.imshow("image", large_img)
                print("Press any key to continue...")
                cv2.waitKey(0)
            else:
                print(red("\nYou need to load an image first."))
        elif option == 4:
            cv2.destroyAllWindows()
            break
        else:
            print(red("\nInvalid option, please try again."))


def get_rectangle_from_word(word_to_rect, target_word):
    if target_word in word_to_rect:
        return word_to_rect[target_word]
    else:
        return None


def assign_characters_to_words(word_rects, char_rects):
    char_to_word = {}

    for char in char_rects:
        for word in word_rects:

            if word[0] <= char[0] and word[1] <= char[1] and word[0] + word[2] >= char[0] + char[2] and \
                    word[1] + word[3] >= char[1] + char[3]:
                char_to_word[tuple(char)] = tuple(word)
                break

    return char_to_word


def openImageFindContours():
    image_file_name = input("\nPlease enter the image file name: ")
    image_path = f'../Images/{image_file_name}'

    if not os.path.exists(image_path):
        print(red(f"The file {image_file_name} does not exist.\n"))
        return None, None

    large_img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if large_img_gray is None:
        print(red(f"The file {image_file_name} could not be opened.\n"))
        return None, None

    large_img = cv2.cvtColor(large_img_gray, cv2.COLOR_GRAY2BGR)

    _, binary_img = cv2.threshold(large_img_gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, large_img


def find_letters_rect(contours):
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rects.append([x, y, x + w, y + h])

    rects = np.array(rects)
    return rects


def find_word_rects(letter_rects):
    global global_rectangles
    rectangles = []
    clustering = DBSCAN(eps=60, min_samples=1).fit(letter_rects)
    for idx, class_ in enumerate(set(clustering.labels_)):
        if class_ != -1:
            same_group = np.array(letter_rects)[np.where(clustering.labels_ == class_)[0]]
            x_min = np.min(same_group[:, 0])
            y_min = np.min(same_group[:, 1])
            x_max = np.max(same_group[:, 2])
            y_max = np.max(same_group[:, 3])
            rectangles.append((x_min, y_min, x_max, y_max))
    global_rectangles = rectangles
    return rectangles


def find_rectangle(point):
    rectangles = global_rectangles
    x, y = point
    for idx, rect in enumerate(rectangles):
        x_min, y_min, x_max, y_max = rect
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return rect, idx
    return None, None


def click_and_crop(event, x, y, flags, param):
    global global_rectangles, selected_indices, current_status
    image = param[0]
    dict2 = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        rect, rect_index = find_rectangle((x, y))

        if rect is not None and rect_index not in selected_indices:
            current_status = rect_to_position[rect]

            selected_indices.append(rect_index)
            cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow("image", image)

            if rect in dict2:
                del dict2[rect]


def combine_letters_into_words(word_to_chars, char_to_image):
    word_to_string = {}
    for word_rect, char_rects in word_to_chars.items():
        word = ''.join(char_to_image[char_rect] for char_rect in char_rects)
        word_to_string[word_rect] = word
    return word_to_string


def crop_letters_from_image(contours, large_img):
    dict = {}
    letters = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        rect = [x, y, x + w, y + h]
        letters.append(rect)
        margin = 1

        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(large_img.shape[1], x + w + margin)
        y_end = min(large_img.shape[0], y + h + margin)
        cropped_img = large_img[y_start:y_end, x_start:x_end]

        pil_img = Image.fromarray(cropped_img).resize((28, 28))
        pil_img = ImageOps.invert(pil_img)
        pil_img.save('letters/letter_{}.png'.format(idx))
        img = transform(pil_img).unsqueeze(0)
        model.eval()

        with torch.no_grad():
            output = model(img)

        _, predicted = torch.max(output, 1)
        predicted_label = chr(predicted.item() + ord('a') - 1)
        dict[tuple(rect)] = predicted_label

    return dict


def generate_words(letter_to_word, letter_to_char):
    word_to_letters = {}
    for letter_rect, word_rect in letter_to_word.items():
        if word_rect in word_to_letters:
            word_to_letters[word_rect].append((letter_rect, letter_to_char[letter_rect]))
        else:
            word_to_letters[word_rect] = [(letter_rect, letter_to_char[letter_rect])]

    for word_rect, letters in word_to_letters.items():
        letters.sort(key=lambda x: x[0][0])

    word_to_string = {word_rect: ''.join([letter[1] for letter in letters]) for word_rect, letters in
                      word_to_letters.items()}

    return word_to_string


def find_word_containing_char(word_rects, char_rect):
    word_rect_distances = []
    for word_rect in word_rects:
        if word_rect[0] <= char_rect[0] and word_rect[1] <= char_rect[1] and \
                word_rect[2] >= char_rect[2] and word_rect[3] >= char_rect[3]:
            word_center = ((word_rect[2] - word_rect[0]) / 2, (word_rect[3] - word_rect[1]) / 2)
            char_center = ((char_rect[2] - char_rect[0]) / 2, (char_rect[3] - char_rect[1]) / 2)
            distance = ((word_center[0] - char_center[0]) ** 2 + (word_center[1] - char_center[1]) ** 2) ** 0.5
            word_rect_distances.append((distance, word_rect))

    word_rect_distances.sort()
    return word_rect_distances[0][1] if word_rect_distances else None


def map_rects_to_words(word_rects, letter_rects):
    rect_to_word = {}
    for letter_rect in letter_rects:
        word_rect = find_word_containing_char(word_rects, letter_rect)
        if word_rect is not None:
            rect_to_word[tuple(letter_rect)] = word_rect
    return rect_to_word


def find_matching_rect(word_to_rect, target_string):
    matching_rects = [rect for rect, word in word_to_rect.items() if word == target_string]

    for rect in matching_rects:
        position_info = rect_to_position.get(rect, {})
        line_num = position_info.get('line', 0)
        pos_num = position_info.get('position', 0)

        if line_num >= current_status['line']:
            if line_num > current_status['line'] or pos_num > current_status['position']:
                return rect

    return None


def reverse_dict_order(input_dict):
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    reversed_dict = dict(zip(keys[::-1], values[::-1]))
    return reversed_dict


def map_rectangles_to_lines_and_positions(rectangles):
    lines = {}

    rectangles.sort(key=lambda x: x[1])
    current_line = 1
    current_y = rectangles[0][1]
    current_line_rectangles = []
    for rect in rectangles:
        if abs(rect[1] - current_y) < 30:
            current_line_rectangles.append(rect)
        else:
            lines[current_y] = sorted(current_line_rectangles,
                                      key=lambda x: x[0])
            current_y = rect[1]
            current_line_rectangles = [rect]
            current_line += 1
    if current_line_rectangles:
        lines[current_y] = sorted(current_line_rectangles, key=lambda x: x[0])

    result = {}
    for line_index, line_rectangles in enumerate(lines.values(), start=1):
        for position, rect in enumerate(line_rectangles, start=1):
            result[tuple(rect)] = {'line': line_index, 'position': position}

    return result
