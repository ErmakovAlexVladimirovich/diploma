# -*- coding: utf-8 -*-
import json
import math
import numpy as np
import os.path
import csv
import openpyxl

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import cv2


def is_coordinate_inside_circle(x, y, center_x, center_y, radius):
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance <= radius


def rotate_template_and_get_mask(template, angle):
    # Get the dimensions of the image
    height, width = template.shape[:2]

    # Calculate the new dimensions to accommodate the rotated image
    new_width = int(np.ceil(width * np.abs(np.cos(np.radians(angle)))) + np.ceil(height * np.abs(np.sin(np.radians(angle)))))
    new_height = int(np.ceil(height * np.abs(np.cos(np.radians(angle)))) + np.ceil(width * np.abs(np.sin(np.radians(angle)))))

    # Calculate the center of the image
    center_x = int(width / 2)
    center_y = int(height / 2)

    # Calculate the center of the new canvas
    new_center_x = int(new_width / 2)
    new_center_y = int(new_height / 2)

    # Calculate the translation required to center the image on the new canvas
    tx = new_center_x - center_x
    ty = new_center_y - center_y

    # Calculate the transformation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Adjust the translation component of the matrix
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty

    # Apply the rotation to the image
    rotated_template = cv2.warpAffine(template, rotation_matrix, (new_width, new_height))
    rotated_mask = cv2.warpAffine(cv2.bitwise_not(template), rotation_matrix, (new_width, new_height))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    rotated_mask = cv2.dilate(rotated_mask, kernel, iterations=1)
    # ret, rotated_mask = cv2.threshold(rotated_mask, 10, 255, cv2.THRESH_BINARY)
    cv2.normalize(rotated_mask, rotated_mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return rotated_template, rotated_mask


def find_template_variations(image, template, furniture_name, elements_amount, threshold=0.96):

    # Preprocess image and template
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, image = cv2.threshold(image_gray, 125, 255, cv2.THRESH_BINARY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    template_height, template_width = template.shape

    # Set a threshold for template matching results
    threshold = threshold
    rotation_angles = np.arange(0, 360, 5)  # Rotation angles to consider

    # Find template variations
    template_variations = []

    for angle in rotation_angles:
        # Rotate the template
        rotated_template, rotated_mask = rotate_template_and_get_mask(template, angle)

        # Get template width and height
        rotated_template_height, rotated_template_width = rotated_template.shape

        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCORR_NORMED, mask=rotated_mask)
        loc = np.where(result >= threshold)
        local_threshold = threshold
        while len(loc[0]) > elements_amount*2:
            local_threshold += 0.01
            loc = np.where(result >= local_threshold)

        for pt in zip(*loc[::-1]):
            # Calculate the bounding box coordinates
            center = int(pt[0] + rotated_template_width / 2), int(pt[1] + rotated_template_height / 2)
            variation = {
                'center': center,
                'size': (template_width, template_height),
                # 'scale': scale,
                'rotation_angle': angle,
                'furniture_name': furniture_name
            }
            template_variations.append(variation)

    margin = 5

    filtered_array = []
    for i, dict1 in enumerate(template_variations):
        is_close = False
        for j, dict2 in enumerate(template_variations[i + 1:]):
            if abs(dict1['center'][0] - dict2['center'][0]) <= margin and abs(
                    dict1['center'][1] - dict2['center'][1]) <= margin:
                is_close = True
                break
        if not is_close:
            filtered_array.append(dict1)

    template_variations = filtered_array

    print(f'Were found {len(template_variations)} variations of {furniture_name}')
    return template_variations


def is_corner_in_walls(walls, corner):
    for wall in walls:
        if corner == wall[0] or corner == wall[1]:
            return True
    return False


def is_point_in_found_elements(elements, point):
    if elements is not None:
        for found_element in elements:
            current_rotated_rectangle = RotatedRectangle(found_element['center'], found_element['size'],
                                                         found_element['rotation_angle'])
            if current_rotated_rectangle.is_point_inside(point):
                return found_element
    return None


def import_csv_config():
    with open('application-config.csv', 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        furniture_list = [row for row in reader]
    return furniture_list


def import_xlsx_config():
    workbook = openpyxl.load_workbook('application-config.xlsx')
    sheet = workbook.worksheets[0]
    furniture_list = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        furniture_data = {
            'Тип мебели': row[0],
            'Отображаемое в приложении название': row[1],
            'Название объекта конструктора в Blender': row[2],
        }
        furniture_list.append(furniture_data)
    workbook.close()
    return furniture_list


class Ui_MainWindow(object):

    def __init__(self):
        super().__init__()

        self.KERNEL_SPIN_BOX_MIN_VAL = 2
        self.KERNEL_SPIN_BOX_MAX_VAL = 15
        self.KERNEL_SPIN_BOX_STEP = 1
        self.IMAGE_UPSCALE_RATE = 3
        self.SELECTING_TOLERANCE = 5

        self.furniture_list = import_xlsx_config()
        self.setup_ui(MainWindow)

        self.image_path = None
        self.image = None
        self.initial_image = None

        # Variables for drawing
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.colors = [self.GREEN, self.YELLOW, self.RED]


        self.image_label.setMouseTracking(True)
        self.middle_button_pressed = False
        self.middle_button_start_pos = QtCore.QPoint()
        self.middle_button_start_hscroll_pos = 0
        self.middle_button_start_vscroll_pos = 0

        self.image_with_corners = None
        self.filtered_image = None
        self.filtered_image_with_corners = None
        self.image_with_rectangle = None
        self.corners_found = False
        self.corners = []
        self.confirmed_corners = []
        self.corners_confirmed = False
        self.first_selected_corner = None
        self.walls = []
        self.outside_walls = []
        self.confirmed_walls = []
        self.threshold_current_value = 0
        self.dilate_1 = False
        self.erode_1 = False
        self.scale_factor = 1
        self.horizontal_image_size_scale = 1.0
        self.vertical_image_size_scale = 1.0
        self.real_sizes_are_shown = False
        self.image_real_sizes_scale_calculated = False
        self.real_sizes_confirmed = False
        self.scale_changed = None

        # Variables for drawing rectangles
        self.drawing = False
        self.rectangle = None
        self.rectangle_drawn = False
        self.start_x, self.start_y = -1, -1
        self.current_x, self.current_y = -1, -1

        # Variables for moving rectangle
        self.top_left_delta_x = -1
        self.top_left_delta_y = -1
        self.bottom_right_delta_x = -1
        self.bottom_right_delta_y = -1
        self.roi_center_delta = [-1, -1]

        # Variables for resizing rectangle
        self.start_pos = None
        self.end_pos = None
        self.resizing_rectangle = False
        self.moving_roi = False
        self.resizing_corner = None

        # Variables for interior elements search
        self.image_with_interior_elements = None
        self.region_of_interest = None
        self.found_windows = None
        self.confirmed_windows = None
        self.found_doors = None
        self.confirmed_doors = None
        self.found_furniture = None
        self.confirmed_furniture = None
        self.draw_region_of_interest = False

        # Exporting data
        self.adjusted_walls = []
        self.json_data = None


    def setup_ui(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1792, 1044)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter_2 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter = QtWidgets.QSplitter(self.splitter_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.scrollArea = QtWidgets.QScrollArea(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1224, 1132))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.image_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setText("")
        # self.image_label.setPixmap(QtGui.QPixmap(".\\GUI\\../Images/flat-plan-test-2.jpg"))
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setObjectName("image_label")
        self.gridLayout_2.addWidget(self.image_label, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.scale_slider_label = QtWidgets.QLabel(self.widget)
        self.scale_slider_label.setMinimumSize(QtCore.QSize(90, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.scale_slider_label.setFont(font)
        self.scale_slider_label.setStyleSheet("")
        self.scale_slider_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.scale_slider_label.setObjectName("scale_slider_label")
        self.horizontalLayout.addWidget(self.scale_slider_label)
        self.scale_slider = QtWidgets.QSlider(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scale_slider.sizePolicy().hasHeightForWidth())
        self.scale_slider.setSizePolicy(sizePolicy)
        self.scale_slider.setMinimumSize(QtCore.QSize(100, 0))
        self.scale_slider.setMaximumSize(QtCore.QSize(400, 16777215))
        self.scale_slider.setStyleSheet("")
        self.scale_slider.setMinimum(50)
        self.scale_slider.setMaximum(400)
        self.scale_slider.setProperty("value", 100)
        self.scale_slider.setOrientation(QtCore.Qt.Horizontal)
        self.scale_slider.setObjectName("scale_slider")
        self.horizontalLayout.addWidget(self.scale_slider)
        self.scale_slider_percent_label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.scale_slider_percent_label.setFont(font)
        self.scale_slider_percent_label.setObjectName("scale_slider_percent_label")
        self.horizontalLayout.addWidget(self.scale_slider_percent_label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)

        self.tabWidget = QtWidgets.QTabWidget(self.splitter_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(525, 0))

        self.tabWidget.setObjectName("tabWidget")
        self.scale_tab = QtWidgets.QWidget()
        self.scale_tab.setObjectName("scale_tab")
        self.tabWidget.addTab(self.scale_tab, "")
        self.corners_tab = QtWidgets.QWidget()
        self.corners_tab.setObjectName("corners_tab")
        self.tabWidget.addTab(self.corners_tab, "")
        self.furniture_tab = QtWidgets.QWidget()
        self.furniture_tab.setObjectName("furniture_tab")
        self.tabWidget.addTab(self.furniture_tab, "")
        self.tabWidget.tabBar().setTabTextColor(0, QtCore.Qt.red)
        self.tabWidget.tabBar().setTabTextColor(1, QtCore.Qt.red)
        self.tabWidget.tabBar().setTabTextColor(2, QtCore.Qt.red)

        self.gridLayout_6 = QtWidgets.QGridLayout(self.scale_tab)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.scrollArea_3 = QtWidgets.QScrollArea(self.scale_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea_3.sizePolicy().hasHeightForWidth())
        self.scrollArea_3.setSizePolicy(sizePolicy)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 544, 967))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_2 = QtWidgets.QFrame(self.scrollAreaWidgetContents_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.frame_2.setFont(font)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setLineWidth(1)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.scale_detection_information_text = QtWidgets.QLabel(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scale_detection_information_text.sizePolicy().hasHeightForWidth())
        self.scale_detection_information_text.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.scale_detection_information_text.setFont(font)
        self.scale_detection_information_text.setFrameShape(QtWidgets.QFrame.Box)
        self.scale_detection_information_text.setTextFormat(QtCore.Qt.AutoText)
        self.scale_detection_information_text.setScaledContents(False)
        self.scale_detection_information_text.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft |
                                                           QtCore.Qt.AlignVCenter)
        self.scale_detection_information_text.setWordWrap(True)
        self.scale_detection_information_text.setObjectName("scale_detection_information_text")
        self.verticalLayout_6.addWidget(self.scale_detection_information_text)
        self.sizes_label = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sizes_label.setFont(font)
        self.sizes_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sizes_label.setObjectName("sizes_label")
        self.verticalLayout_6.addWidget(self.sizes_label)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.horizontal_size_label = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.horizontal_size_label.setFont(font)
        self.horizontal_size_label.setObjectName("horizontal_size_label")
        self.horizontalLayout_20.addWidget(self.horizontal_size_label)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem1)
        self.horizontal_size_spin_box = QtWidgets.QSpinBox(self.frame_2)
        self.horizontal_size_spin_box.setMinimum(40)
        self.horizontal_size_spin_box.setMaximum(40000)
        self.horizontal_size_spin_box.setProperty("value", 400)
        self.horizontal_size_spin_box.setObjectName("horizontal_size_spin_box")
        self.horizontalLayout_20.addWidget(self.horizontal_size_spin_box)
        self.horizontal_size_label_2 = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.horizontal_size_label_2.setFont(font)
        self.horizontal_size_label_2.setObjectName("horizontal_size_label_2")
        self.horizontalLayout_20.addWidget(self.horizontal_size_label_2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_20)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.vertical_size_label = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.vertical_size_label.setFont(font)
        self.vertical_size_label.setObjectName("vertical_size_label")
        self.horizontalLayout_21.addWidget(self.vertical_size_label)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_21.addItem(spacerItem2)
        self.vertical_size_spin_box = QtWidgets.QSpinBox(self.frame_2)
        self.vertical_size_spin_box.setMinimum(40)
        self.vertical_size_spin_box.setMaximum(40000)
        self.vertical_size_spin_box.setProperty("value", 400)
        self.vertical_size_spin_box.setObjectName("vertical_size_spin_box")
        self.horizontalLayout_21.addWidget(self.vertical_size_spin_box)
        self.mm_vertical_label = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.mm_vertical_label.setFont(font)
        self.mm_vertical_label.setObjectName("mm_vertical_label")
        self.horizontalLayout_21.addWidget(self.mm_vertical_label)
        self.verticalLayout_6.addLayout(self.horizontalLayout_21)
        self.calculate_size_push_button = QtWidgets.QPushButton(self.frame_2)
        self.calculate_size_push_button.setObjectName("calculate_size_push_button")
        self.verticalLayout_6.addWidget(self.calculate_size_push_button)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem3)
        self.show_sizes_check_box = QtWidgets.QCheckBox(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.show_sizes_check_box.setFont(font)
        self.show_sizes_check_box.setChecked(False)
        self.show_sizes_check_box.setObjectName("show_sizes_check_box")
        self.horizontalLayout_18.addWidget(self.show_sizes_check_box)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem4)
        self.verticalLayout_6.addLayout(self.horizontalLayout_18)
        self.apply_size_push_button = QtWidgets.QPushButton(self.frame_2)
        self.apply_size_push_button.setObjectName("apply_size_push_button")
        self.verticalLayout_6.addWidget(self.apply_size_push_button)
        spacerItem5 = QtWidgets.QSpacerItem(17, 225, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem5)
        self.gridLayout_5.addWidget(self.frame_2, 0, 0, 1, 1)
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.gridLayout_6.addWidget(self.scrollArea_3, 0, 0, 1, 1)

        self.gridLayout_3 = QtWidgets.QGridLayout(self.corners_tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.corners_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea_2.sizePolicy().hasHeightForWidth())
        self.scrollArea_2.setSizePolicy(sizePolicy)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 507, 940))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.frame = QtWidgets.QFrame(self.scrollAreaWidgetContents_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.frame.setFont(font)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(1)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.corners_detection_information_text = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.corners_detection_information_text.sizePolicy().hasHeightForWidth())
        self.corners_detection_information_text.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.corners_detection_information_text.setFont(font)
        self.corners_detection_information_text.setFrameShape(QtWidgets.QFrame.Box)
        self.corners_detection_information_text.setTextFormat(QtCore.Qt.AutoText)
        self.corners_detection_information_text.setScaledContents(False)
        self.corners_detection_information_text.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.corners_detection_information_text.setWordWrap(True)
        self.corners_detection_information_text.setObjectName("corners_detection_information_text")
        self.verticalLayout.addWidget(self.corners_detection_information_text)
        self.verticalLayout_1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.threshold_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.threshold_label.setFont(font)
        self.threshold_label.setAlignment(QtCore.Qt.AlignCenter)
        self.threshold_label.setObjectName("threshold_label")
        self.verticalLayout_1.addWidget(self.threshold_label)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(-1, -1, 0, -1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem6)
        self.splitter_3 = QtWidgets.QSplitter(self.frame)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.threshold_maxval_label_1 = QtWidgets.QLabel(self.splitter_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.threshold_maxval_label_1.setFont(font)
        self.threshold_maxval_label_1.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.threshold_maxval_label_1.setObjectName("threshold_maxval_label_1")
        self.threshold_maxval = QtWidgets.QSlider(self.splitter_3)
        self.threshold_maxval.setMaximum(255)
        self.threshold_maxval.setProperty("value", 100)
        self.threshold_maxval.setOrientation(QtCore.Qt.Horizontal)
        self.threshold_maxval.setObjectName("threshold_maxval")
        self.threshold_maxval_label_2 = QtWidgets.QLabel(self.splitter_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.threshold_maxval_label_2.setFont(font)
        self.threshold_maxval_label_2.setObjectName("threshold_maxval_label_2")
        self.horizontalLayout_6.addWidget(self.splitter_3)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem6)
        self.verticalLayout_1.addLayout(self.horizontalLayout_6)
        self.verticalLayout.addLayout(self.verticalLayout_1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.erode_1_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.erode_1_label.setFont(font)
        self.erode_1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.erode_1_label.setObjectName("erode_1_label")
        self.verticalLayout_2.addWidget(self.erode_1_label)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.erode_1_check_box = QtWidgets.QCheckBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.erode_1_check_box.setFont(font)
        self.erode_1_check_box.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.erode_1_check_box.setObjectName("erode_1_check_box")
        self.horizontalLayout_5.addWidget(self.erode_1_check_box)
        self.kernel_size_label_2 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.kernel_size_label_2.setFont(font)
        self.kernel_size_label_2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.kernel_size_label_2.setObjectName("kernel_size_label_2")
        self.horizontalLayout_5.addWidget(self.kernel_size_label_2)
        self.erode_1_spin_box = QtWidgets.QSpinBox(self.frame)
        self.erode_1_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.erode_1_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.erode_1_spin_box.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.erode_1_spin_box.setMinimum(self.KERNEL_SPIN_BOX_MIN_VAL)
        self.erode_1_spin_box.setMaximum(self.KERNEL_SPIN_BOX_MAX_VAL)
        self.erode_1_spin_box.setSingleStep(self.KERNEL_SPIN_BOX_STEP)
        self.erode_1_spin_box.setObjectName("erode_1_spin_box")
        self.horizontalLayout_5.addWidget(self.erode_1_spin_box)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.dilate_1_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dilate_1_label.setFont(font)
        self.dilate_1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.dilate_1_label.setObjectName("dilate_1_label")
        self.verticalLayout_3.addWidget(self.dilate_1_label)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.dilate_1_check_box = QtWidgets.QCheckBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dilate_1_check_box.setFont(font)
        self.dilate_1_check_box.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.dilate_1_check_box.setObjectName("dilate_1_check_box")
        self.horizontalLayout_4.addWidget(self.dilate_1_check_box)
        self.kernel_size_label_1 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.kernel_size_label_1.setFont(font)
        self.kernel_size_label_1.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.kernel_size_label_1.setObjectName("kernel_size_label_1")
        self.horizontalLayout_4.addWidget(self.kernel_size_label_1)
        self.dilate_1_spin_box = QtWidgets.QSpinBox(self.frame)
        self.dilate_1_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.dilate_1_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.dilate_1_spin_box.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.dilate_1_spin_box.setMinimum(self.KERNEL_SPIN_BOX_MIN_VAL)
        self.dilate_1_spin_box.setMaximum(self.KERNEL_SPIN_BOX_MAX_VAL)
        self.dilate_1_spin_box.setSingleStep(self.KERNEL_SPIN_BOX_STEP)
        self.dilate_1_spin_box.setObjectName("dilate_1_spin_box")
        self.horizontalLayout_4.addWidget(self.dilate_1_spin_box)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.erode_2_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.erode_2_label.setFont(font)
        self.erode_2_label.setAlignment(QtCore.Qt.AlignCenter)
        self.erode_2_label.setObjectName("erode_2_label")
        self.verticalLayout_5.addWidget(self.erode_2_label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.erode_2_check_box = QtWidgets.QCheckBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.erode_2_check_box.setFont(font)
        self.erode_2_check_box.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.erode_2_check_box.setObjectName("erode_2_check_box")
        self.horizontalLayout_2.addWidget(self.erode_2_check_box)
        self.kernel_size_label_4 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.kernel_size_label_4.setFont(font)
        self.kernel_size_label_4.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.kernel_size_label_4.setObjectName("kernel_size_label_4")
        self.horizontalLayout_2.addWidget(self.kernel_size_label_4)
        self.erode_2_spin_box = QtWidgets.QSpinBox(self.frame)
        self.erode_2_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.erode_2_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.erode_2_spin_box.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.erode_2_spin_box.setMinimum(self.KERNEL_SPIN_BOX_MIN_VAL)
        self.erode_2_spin_box.setMaximum(self.KERNEL_SPIN_BOX_MAX_VAL)
        self.erode_2_spin_box.setSingleStep(self.KERNEL_SPIN_BOX_STEP)
        self.erode_2_spin_box.setObjectName("erode_2_spin_box")
        self.horizontalLayout_2.addWidget(self.erode_2_spin_box)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.verticalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.dilate_2_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dilate_2_label.setFont(font)
        self.dilate_2_label.setAlignment(QtCore.Qt.AlignCenter)
        self.dilate_2_label.setObjectName("dilate_2_label")
        self.verticalLayout_4.addWidget(self.dilate_2_label)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.dilate_2_check_box = QtWidgets.QCheckBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.dilate_2_check_box.setFont(font)
        self.dilate_2_check_box.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.dilate_2_check_box.setObjectName("dilate_2_check_box")
        self.horizontalLayout_3.addWidget(self.dilate_2_check_box)
        self.kernel_size_label_3 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.kernel_size_label_3.setFont(font)
        self.kernel_size_label_3.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.kernel_size_label_3.setObjectName("kernel_size_label_3")
        self.horizontalLayout_3.addWidget(self.kernel_size_label_3)
        self.dilate_2_spin_box = QtWidgets.QSpinBox(self.frame)
        self.dilate_2_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.dilate_2_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.dilate_2_spin_box.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.dilate_2_spin_box.setMinimum(self.KERNEL_SPIN_BOX_MIN_VAL)
        self.dilate_2_spin_box.setMaximum(self.KERNEL_SPIN_BOX_MAX_VAL)
        self.dilate_2_spin_box.setSingleStep(self.KERNEL_SPIN_BOX_STEP)
        self.dilate_2_spin_box.setObjectName("dilate_2_spin_box")
        self.horizontalLayout_3.addWidget(self.dilate_2_spin_box)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addLayout(self.verticalLayout_4)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem6)
        self.show_filters_check_box = QtWidgets.QCheckBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.show_filters_check_box.setFont(font)
        self.show_filters_check_box.setChecked(False)
        self.show_filters_check_box.setObjectName("show_filters_check_box")
        self.horizontalLayout_7.addWidget(self.show_filters_check_box)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem7)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.max_corners_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.max_corners_label.setFont(font)
        self.max_corners_label.setObjectName("max_corners_label")
        self.horizontalLayout_8.addWidget(self.max_corners_label)
        spacerItem8 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem8)
        self.max_corners_spin_box = QtWidgets.QSpinBox(self.frame)
        self.max_corners_spin_box.setMinimum(0)
        self.max_corners_spin_box.setMaximum(200)
        self.max_corners_spin_box.setProperty("value", 40)
        self.max_corners_spin_box.setObjectName("max_corners_spin_box")
        self.horizontalLayout_8.addWidget(self.max_corners_spin_box)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.quality_level_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.quality_level_label.setFont(font)
        self.quality_level_label.setObjectName("quality_level_label")
        self.horizontalLayout_10.addWidget(self.quality_level_label)
        spacerItem9 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem9)
        self.quality_level_spin_box = QtWidgets.QDoubleSpinBox(self.frame)
        self.quality_level_spin_box.setMaximum(0.99)
        self.quality_level_spin_box.setSingleStep(0.05)
        self.quality_level_spin_box.setProperty("value", 0.5)
        self.quality_level_spin_box.setObjectName("quality_level_spin_box")
        self.horizontalLayout_10.addWidget(self.quality_level_spin_box)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.min_distance_label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.min_distance_label.setFont(font)
        self.min_distance_label.setObjectName("min_distance_label")
        self.horizontalLayout_9.addWidget(self.min_distance_label)
        spacerItem10 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem10)
        self.min_distance_spin_box = QtWidgets.QSpinBox(self.frame)
        self.min_distance_spin_box.setMinimum(0)
        self.min_distance_spin_box.setMaximum(200)
        self.min_distance_spin_box.setProperty("value", 5)
        self.min_distance_spin_box.setObjectName("min_distance_spin_box")
        self.horizontalLayout_9.addWidget(self.min_distance_spin_box)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.find_corners_push_button = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.find_corners_push_button.sizePolicy().hasHeightForWidth())
        self.find_corners_push_button.setSizePolicy(sizePolicy)
        self.find_corners_push_button.setObjectName("find_corners_push_button")
        self.verticalLayout.addWidget(self.find_corners_push_button)
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem11)
        self.filter_found_corners_label = QtWidgets.QLabel(self.frame)
        self.filter_found_corners_label.setObjectName("label_3")
        self.horizontalLayout_12.addWidget(self.filter_found_corners_label)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem7)
        self.verticalLayout.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.confirm_corner_radio_button = QtWidgets.QRadioButton(self.frame)
        self.confirm_corner_radio_button.setChecked(True)
        self.confirm_corner_radio_button.setObjectName("confirm_corner_radio_button")
        self.horizontalLayout_11.addWidget(self.confirm_corner_radio_button)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem12)
        self.add_corner_radio_button = QtWidgets.QRadioButton(self.frame)
        self.add_corner_radio_button.setObjectName("add_corner_radio_button")
        self.horizontalLayout_11.addWidget(self.add_corner_radio_button)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem12)
        self.delete_corner_radio_button = QtWidgets.QRadioButton(self.frame)
        self.delete_corner_radio_button.setObjectName("delete_corner_radio_button")
        self.horizontalLayout_11.addWidget(self.delete_corner_radio_button)
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.confirm_corners_push_button = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.confirm_corners_push_button.sizePolicy().hasHeightForWidth())
        self.confirm_corners_push_button.setSizePolicy(sizePolicy)
        self.confirm_corners_push_button.setObjectName("confirm_corners_push_button")
        self.verticalLayout.addWidget(self.confirm_corners_push_button)
        self.line_3 = QtWidgets.QFrame(self.frame)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout.addWidget(self.line_3)
        self.wall_confirmation_text = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.wall_confirmation_text.sizePolicy().hasHeightForWidth())
        self.wall_confirmation_text.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.wall_confirmation_text.setFont(font)
        self.wall_confirmation_text.setFrameShape(QtWidgets.QFrame.Box)
        self.wall_confirmation_text.setTextFormat(QtCore.Qt.AutoText)
        self.wall_confirmation_text.setScaledContents(False)
        self.wall_confirmation_text.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.wall_confirmation_text.setWordWrap(True)
        self.wall_confirmation_text.setObjectName("wall_confirmation_text")
        self.verticalLayout.addWidget(self.wall_confirmation_text)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.add_wall_radio_button = QtWidgets.QRadioButton(self.frame)
        self.add_wall_radio_button.setObjectName("add_wall_radio_button")
        self.horizontalLayout_13.addWidget(self.add_wall_radio_button)
        self.horizontalLayout_13.addItem(spacerItem13)
        self.delete_wall_radio_button = QtWidgets.QRadioButton(self.frame)
        self.delete_wall_radio_button.setObjectName("delete_wall_radio_button")
        self.horizontalLayout_13.addWidget(self.delete_wall_radio_button)
        self.verticalLayout.addLayout(self.horizontalLayout_13)
        self.confirm_walls_push_button = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.confirm_walls_push_button.sizePolicy().hasHeightForWidth())
        self.confirm_walls_push_button.setSizePolicy(sizePolicy)
        self.confirm_walls_push_button.setObjectName("confirm_walls_push_button")
        self.verticalLayout.addWidget(self.confirm_walls_push_button)
        spacerItem14 = QtWidgets.QSpacerItem(17, 17, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem14)
        self.gridLayout_4.addWidget(self.frame, 0, 0, 1, 1)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_3.addWidget(self.scrollArea_2, 0, 1, 1, 1)

        self.gridLayout_8 = QtWidgets.QGridLayout(self.furniture_tab)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.scrollArea_4 = QtWidgets.QScrollArea(self.furniture_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea_4.sizePolicy().hasHeightForWidth())
        self.scrollArea_4.setSizePolicy(sizePolicy)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollArea_4.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.scrollArea_4.setObjectName("scrollArea_4")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 527, 971))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.frame_3 = QtWidgets.QFrame(self.scrollAreaWidgetContents_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.frame_3.setFont(font)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setLineWidth(1)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.corners_detection_information_text_2 = QtWidgets.QLabel(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.corners_detection_information_text_2.sizePolicy().hasHeightForWidth())
        self.corners_detection_information_text_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.corners_detection_information_text_2.setFont(font)
        self.corners_detection_information_text_2.setFrameShape(QtWidgets.QFrame.Box)
        self.corners_detection_information_text_2.setTextFormat(QtCore.Qt.AutoText)
        self.corners_detection_information_text_2.setScaledContents(False)
        self.corners_detection_information_text_2.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.corners_detection_information_text_2.setWordWrap(True)
        self.corners_detection_information_text_2.setObjectName("corners_detection_information_text_2")
        self.verticalLayout_7.addWidget(self.corners_detection_information_text_2)
        self.roi_label = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.roi_label.setFont(font)
        self.roi_label.setAlignment(QtCore.Qt.AlignCenter)
        self.roi_label.setObjectName("roi_label")
        self.verticalLayout_7.addWidget(self.roi_label)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.roi_rotation_label_1 = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.roi_rotation_label_1.setFont(font)
        self.roi_rotation_label_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.roi_rotation_label_1.setObjectName("roi_rotation_label_1")
        self.horizontalLayout_14.addWidget(self.roi_rotation_label_1)
        spacerItem18 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem18)
        self.roi_rotation_label_2 = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.roi_rotation_label_2.setFont(font)
        self.roi_rotation_label_2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.roi_rotation_label_2.setObjectName("roi_rotation_label_2")
        self.horizontalLayout_14.addWidget(self.roi_rotation_label_2)
        self.roi_rotation_slider = QtWidgets.QSlider(self.frame_3)
        self.roi_rotation_slider.setMinimumSize(QtCore.QSize(359, 0))
        self.roi_rotation_slider.setMaximum(359)
        self.roi_rotation_slider.setProperty("value", 0)
        self.roi_rotation_slider.setOrientation(QtCore.Qt.Horizontal)
        self.roi_rotation_slider.setObjectName("roi_rotation_slider")
        self.horizontalLayout_14.addWidget(self.roi_rotation_slider)
        self.roi_rotation_label_3 = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.roi_rotation_label_3.setFont(font)
        self.roi_rotation_label_3.setObjectName("roi_rotation_label_3")
        self.horizontalLayout_14.addWidget(self.roi_rotation_label_3)
        self.verticalLayout_7.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.roi_width_control_label = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.roi_width_control_label.setFont(font)
        self.roi_width_control_label.setAlignment(QtCore.Qt.AlignCenter)
        self.roi_width_control_label.setObjectName("roi_width_control_label")
        self.horizontalLayout_24.addWidget(self.roi_width_control_label)
        self.decrease_roi_width_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.decrease_roi_width_push_button.sizePolicy().hasHeightForWidth())
        self.decrease_roi_width_push_button.setSizePolicy(sizePolicy)
        self.decrease_roi_width_push_button.setMaximumSize(QtCore.QSize(40, 40))
        self.decrease_roi_width_push_button.setObjectName("decrease_roi_width_push_button")
        self.horizontalLayout_24.addWidget(self.decrease_roi_width_push_button)
        self.increase_roi_width_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.increase_roi_width_push_button.sizePolicy().hasHeightForWidth())
        self.increase_roi_width_push_button.setSizePolicy(sizePolicy)
        self.increase_roi_width_push_button.setMaximumSize(QtCore.QSize(40, 40))
        self.increase_roi_width_push_button.setObjectName("increase_roi_width_push_button")
        self.horizontalLayout_24.addWidget(self.increase_roi_width_push_button)

        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_24.addItem(spacerItem19)

        self.add_region_of_interest_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.add_region_of_interest_radio_button.setObjectName("add_region_of_interest_radio_button")
        self.add_region_of_interest_radio_button.setChecked(True)
        self.horizontalLayout_24.addWidget(self.add_region_of_interest_radio_button)

        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_24.addItem(spacerItem19)
        self.roi_height_control_label = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.roi_height_control_label.setFont(font)
        self.roi_height_control_label.setAlignment(QtCore.Qt.AlignCenter)
        self.roi_height_control_label.setObjectName("roi_height_control_label")
        self.horizontalLayout_24.addWidget(self.roi_height_control_label)
        self.decrease_roi_height_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.decrease_roi_height_push_button.sizePolicy().hasHeightForWidth())
        self.decrease_roi_height_push_button.setSizePolicy(sizePolicy)
        self.decrease_roi_height_push_button.setMaximumSize(QtCore.QSize(40, 40))
        self.decrease_roi_height_push_button.setObjectName("decrease_roi_height_push_button")
        self.horizontalLayout_24.addWidget(self.decrease_roi_height_push_button)
        self.increase_roi_height_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.increase_roi_height_push_button.sizePolicy().hasHeightForWidth())
        self.increase_roi_height_push_button.setSizePolicy(sizePolicy)
        self.increase_roi_height_push_button.setMaximumSize(QtCore.QSize(40, 40))
        self.increase_roi_height_push_button.setObjectName("increase_roi_height_push_button")
        self.horizontalLayout_24.addWidget(self.increase_roi_height_push_button)
        self.verticalLayout_7.addLayout(self.horizontalLayout_24)

        self.line_6 = QtWidgets.QFrame(self.frame_3)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout_7.addWidget(self.line_6)
        self.windows_label = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.windows_label.setFont(font)
        self.windows_label.setAlignment(QtCore.Qt.AlignCenter)
        self.windows_label.setObjectName("windows_label")
        self.verticalLayout_7.addWidget(self.windows_label)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.window_casements_label = QtWidgets.QLabel(self.frame_3)
        self.window_casements_label.setObjectName("window_casements_label")
        self.horizontalLayout_16.addWidget(self.window_casements_label)
        spacerItem21 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem21)
        self.windows_casements_spin_box = QtWidgets.QSpinBox(self.frame_3)
        self.windows_casements_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.windows_casements_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.windows_casements_spin_box.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.windows_casements_spin_box.setMinimum(1)
        self.windows_casements_spin_box.setMaximum(10)
        self.windows_casements_spin_box.setSingleStep(1)
        self.windows_casements_spin_box.setObjectName("windows_casements_spin_box")
        self.horizontalLayout_16.addWidget(self.windows_casements_spin_box)
        self.verticalLayout_7.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.windows_to_find_label = QtWidgets.QLabel(self.frame_3)
        self.windows_to_find_label.setObjectName("windows_to_find_label")
        self.horizontalLayout_17.addWidget(self.windows_to_find_label)
        spacerItem22 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem22)
        self.windows_to_find_spin_box = QtWidgets.QSpinBox(self.frame_3)
        self.windows_to_find_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.windows_to_find_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.windows_to_find_spin_box.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.windows_to_find_spin_box.setMinimum(1)
        self.windows_to_find_spin_box.setMaximum(40)
        self.windows_to_find_spin_box.setSingleStep(1)
        self.windows_to_find_spin_box.setObjectName("windows_to_find_spin_box")
        self.horizontalLayout_17.addWidget(self.windows_to_find_spin_box)
        self.verticalLayout_7.addLayout(self.horizontalLayout_17)
        self.find_windows_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.find_windows_push_button.sizePolicy().hasHeightForWidth())
        self.find_windows_push_button.setSizePolicy(sizePolicy)
        self.find_windows_push_button.setObjectName("find_windows_push_button")
        self.verticalLayout_7.addWidget(self.find_windows_push_button)
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.confirm_window_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.confirm_window_radio_button.setObjectName("confirm_window_radio_button")
        self.horizontalLayout_27.addWidget(self.confirm_window_radio_button)
        spacerItem23 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_27.addItem(spacerItem23)
        self.add_window_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.add_window_radio_button.setObjectName("add_window_radio_button")
        self.horizontalLayout_27.addWidget(self.add_window_radio_button)
        spacerItem24 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_27.addItem(spacerItem24)
        self.delete_window_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.delete_window_radio_button.setObjectName("delete_window_radio_button")
        self.horizontalLayout_27.addWidget(self.delete_window_radio_button)
        self.verticalLayout_7.addLayout(self.horizontalLayout_27)
        self.confirm_windows_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.confirm_windows_push_button.sizePolicy().hasHeightForWidth())
        self.confirm_windows_push_button.setSizePolicy(sizePolicy)
        self.confirm_windows_push_button.setObjectName("confirm_windows_push_button")
        self.verticalLayout_7.addWidget(self.confirm_windows_push_button)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        spacerItem25 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_22.addItem(spacerItem25)
        self.show_confired_windows_ckeck_box = QtWidgets.QCheckBox(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.show_confired_windows_ckeck_box.setFont(font)
        self.show_confired_windows_ckeck_box.setChecked(False)
        self.show_confired_windows_ckeck_box.setObjectName("show_confired_windows_ckeck_box")
        self.horizontalLayout_22.addWidget(self.show_confired_windows_ckeck_box)
        spacerItem26 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_22.addItem(spacerItem26)
        self.verticalLayout_7.addLayout(self.horizontalLayout_22)
        self.line_4 = QtWidgets.QFrame(self.frame_3)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout_7.addWidget(self.line_4)
        self.doors_label = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.doors_label.setFont(font)
        self.doors_label.setAlignment(QtCore.Qt.AlignCenter)
        self.doors_label.setObjectName("doors_label")
        self.verticalLayout_7.addWidget(self.doors_label)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.door_type_label = QtWidgets.QLabel(self.frame_3)
        self.door_type_label.setObjectName("door_type_label")
        self.horizontalLayout_15.addWidget(self.door_type_label)
        spacerItem27 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem27)
        self.door_type_combo_box = QtWidgets.QComboBox(self.frame_3)
        self.door_type_combo_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.door_type_combo_box.setObjectName("door_type_combo_box")
        self.door_type_combo_box.addItem("")
        self.door_type_combo_box.addItem("")
        self.door_type_combo_box.addItem("")
        self.horizontalLayout_15.addWidget(self.door_type_combo_box)
        self.verticalLayout_7.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.doors_to_find_label = QtWidgets.QLabel(self.frame_3)
        self.doors_to_find_label.setObjectName("doors_to_find_label")
        self.horizontalLayout_19.addWidget(self.doors_to_find_label)
        spacerItem28 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_19.addItem(spacerItem28)
        self.doors_to_find_spin_box = QtWidgets.QSpinBox(self.frame_3)
        self.doors_to_find_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.doors_to_find_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.doors_to_find_spin_box.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.doors_to_find_spin_box.setMinimum(1)
        self.doors_to_find_spin_box.setMaximum(40)
        self.doors_to_find_spin_box.setSingleStep(1)
        self.doors_to_find_spin_box.setObjectName("doors_to_find_spin_box")
        self.horizontalLayout_19.addWidget(self.doors_to_find_spin_box)
        self.verticalLayout_7.addLayout(self.horizontalLayout_19)
        self.find_doors_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.find_doors_push_button.sizePolicy().hasHeightForWidth())
        self.find_doors_push_button.setSizePolicy(sizePolicy)
        self.find_doors_push_button.setCheckable(False)
        self.find_doors_push_button.setObjectName("find_doors_push_button")
        self.verticalLayout_7.addWidget(self.find_doors_push_button)
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.confirm_door_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.confirm_door_radio_button.setObjectName("confirm_door_radio_button")
        self.horizontalLayout_29.addWidget(self.confirm_door_radio_button)
        spacerItem29 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_29.addItem(spacerItem29)
        self.add_door_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.add_door_radio_button.setObjectName("add_door_radio_button")
        self.horizontalLayout_29.addWidget(self.add_door_radio_button)
        spacerItem30 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_29.addItem(spacerItem30)
        self.delete_door_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.delete_door_radio_button.setObjectName("delete_door_radio_button")
        self.horizontalLayout_29.addWidget(self.delete_door_radio_button)
        self.verticalLayout_7.addLayout(self.horizontalLayout_29)
        self.confirm_doors_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.confirm_doors_push_button.sizePolicy().hasHeightForWidth())
        self.confirm_doors_push_button.setSizePolicy(sizePolicy)
        self.confirm_doors_push_button.setObjectName("confirm_doors_push_button")
        self.verticalLayout_7.addWidget(self.confirm_doors_push_button)
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        spacerItem31 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_23.addItem(spacerItem31)
        self.show_confired_doors_ckeck_box = QtWidgets.QCheckBox(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.show_confired_doors_ckeck_box.setFont(font)
        self.show_confired_doors_ckeck_box.setChecked(False)
        self.show_confired_doors_ckeck_box.setObjectName("show_confired_doors_ckeck_box")
        self.horizontalLayout_23.addWidget(self.show_confired_doors_ckeck_box)
        spacerItem32 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_23.addItem(spacerItem32)
        self.verticalLayout_7.addLayout(self.horizontalLayout_23)
        self.line_5 = QtWidgets.QFrame(self.frame_3)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout_7.addWidget(self.line_5)
        self.furniture_label = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.furniture_label.setFont(font)
        self.furniture_label.setAlignment(QtCore.Qt.AlignCenter)
        self.furniture_label.setObjectName("furniture_label")
        self.verticalLayout_7.addWidget(self.furniture_label)
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.furniture_type_label = QtWidgets.QLabel(self.frame_3)
        self.furniture_type_label.setObjectName("furniture_type_label")
        self.horizontalLayout_26.addWidget(self.furniture_type_label)
        spacerItem33 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_26.addItem(spacerItem33)
        self.furniture_type_combo_box = QtWidgets.QComboBox(self.frame_3)
        self.furniture_type_combo_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.furniture_type_combo_box.setObjectName("furniture_type_combo_box")

        for furniture in self.furniture_list:
            self.furniture_type_combo_box.addItem("")

        self.horizontalLayout_26.addWidget(self.furniture_type_combo_box)
        self.verticalLayout_7.addLayout(self.horizontalLayout_26)
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.furniture_to_find_label = QtWidgets.QLabel(self.frame_3)
        self.furniture_to_find_label.setObjectName("furniture_to_find_label")
        self.horizontalLayout_28.addWidget(self.furniture_to_find_label)
        spacerItem34 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_28.addItem(spacerItem34)
        self.furniture_to_find_spin_box = QtWidgets.QSpinBox(self.frame_3)
        self.furniture_to_find_spin_box.setMinimumSize(QtCore.QSize(32, 0))
        self.furniture_to_find_spin_box.setMaximumSize(QtCore.QSize(62, 22))
        self.furniture_to_find_spin_box.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.furniture_to_find_spin_box.setMinimum(1)
        self.furniture_to_find_spin_box.setMaximum(40)
        self.furniture_to_find_spin_box.setSingleStep(1)
        self.furniture_to_find_spin_box.setObjectName("furniture_to_find_spin_box")
        self.horizontalLayout_28.addWidget(self.furniture_to_find_spin_box)
        self.verticalLayout_7.addLayout(self.horizontalLayout_28)
        self.find_furniture_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.find_furniture_push_button.sizePolicy().hasHeightForWidth())
        self.find_furniture_push_button.setSizePolicy(sizePolicy)
        self.find_furniture_push_button.setObjectName("find_furniture_push_button")
        self.verticalLayout_7.addWidget(self.find_furniture_push_button)
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.confirm_furniture_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.confirm_furniture_radio_button.setObjectName("confirm_furniture_radio_button")
        self.horizontalLayout_30.addWidget(self.confirm_furniture_radio_button)
        spacerItem35 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_30.addItem(spacerItem35)
        self.add_furniture_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.add_furniture_radio_button.setObjectName("add_furniture_radio_button")
        self.horizontalLayout_30.addWidget(self.add_furniture_radio_button)
        spacerItem36 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_30.addItem(spacerItem36)
        self.delete_furniture_radio_button = QtWidgets.QRadioButton(self.frame_3)
        self.delete_furniture_radio_button.setObjectName("delete_furniture_radio_button")
        self.horizontalLayout_30.addWidget(self.delete_furniture_radio_button)
        self.verticalLayout_7.addLayout(self.horizontalLayout_30)
        self.confirm_furniture_push_button = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.confirm_furniture_push_button.sizePolicy().hasHeightForWidth())
        self.confirm_furniture_push_button.setSizePolicy(sizePolicy)
        self.confirm_furniture_push_button.setObjectName("confirm_furniture_push_button")
        self.verticalLayout_7.addWidget(self.confirm_furniture_push_button)
        spacerItem37 = QtWidgets.QSpacerItem(17, 17, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem37)
        self.gridLayout_7.addWidget(self.frame_3, 0, 0, 1, 1)
        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)
        self.gridLayout_8.addWidget(self.scrollArea_4, 0, 0, 1, 1)

        self.gridLayout.addWidget(self.splitter_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1792, 21))
        self.menubar.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.menubar.setObjectName("menubar")

        self.open_image_action = QtWidgets.QAction("Импорт изображения")
        self.save_data_action = QtWidgets.QAction("Сохранение данных")
        # self.menuOpen_image = QtWidgets.QMenu(self.menubar)
        # self.menuOpen_image.setObjectName("menuOpen_image")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Menubar actions
        self.menubar.addAction(self.open_image_action)
        self.menubar.addAction(self.save_data_action)

        # No size rectangle drawn popup
        self.no_size_rectangle_drawn_box = QtWidgets.QMessageBox()
        self.no_size_rectangle_drawn_box.setIcon(QtWidgets.QMessageBox.Warning)
        self.no_size_rectangle_drawn_box.setWindowTitle('Размерный прямоугольник не создан')
        self.no_size_rectangle_drawn_box.setText('Пожалуйста, создайте прямоугольник, выбрав объект на изображении и указав его границы и размеры. Это необходимо для правильного расчета масштаба изображения. Для создания прямоугольника щелкните и перетащите мышкой по диагонали объекта.')

        self.roi_rotation_angle_label = QtWidgets.QLabel(MainWindow)
        self.roi_rotation_angle_label.setStyleSheet("background-color: white; padding: 5px; border: 1px solid black;")
        self.roi_rotation_angle_label.hide()

        self.retranslate_ui(MainWindow)
        self.setup_connection()
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # MainWindow.setTabOrder(self.tabWidget, self.scrollArea)
        # MainWindow.setTabOrder(self.scrollArea, self.scrollArea_2)
        # MainWindow.setTabOrder(self.scrollArea_2, self.threshold_maxval)
        # MainWindow.setTabOrder(self.threshold_maxval, self.scale_slider)
        # MainWindow.setTabOrder(self.scale_slider, self.erode_1_spin_box)
        # MainWindow.setTabOrder(self.erode_1_spin_box, self.dilate_1_check_box)
        # MainWindow.setTabOrder(self.dilate_1_check_box, self.dilate_1_spin_box)
        # MainWindow.setTabOrder(self.dilate_1_spin_box, self.erode_2_check_box)
        # MainWindow.setTabOrder(self.erode_2_check_box, self.erode_2_spin_box)
        # MainWindow.setTabOrder(self.erode_2_spin_box, self.dilate_2_check_box)
        # MainWindow.setTabOrder(self.dilate_2_check_box, self.dilate_2_spin_box)
        # MainWindow.setTabOrder(self.dilate_2_spin_box, self.show_filters_check_box)
        # MainWindow.setTabOrder(self.show_filters_check_box, self.erode_1_check_box)
        # MainWindow.setTabOrder(self.erode_1_check_box, self.max_corners_spin_box)
        # MainWindow.setTabOrder(self.max_corners_spin_box, self.min_distance_spin_box)
        # MainWindow.setTabOrder(self.min_distance_spin_box, self.quality_level_spin_box)

    def retranslate_ui(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.scale_slider_label.setText(_translate("MainWindow", "Масштаб изображения: 50%"))
        self.scale_slider_percent_label.setText(_translate("MainWindow", "400 %"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.scale_tab), _translate("MainWindow", "Масштаб"))
        self.scale_detection_information_text.setText(_translate("MainWindow",
                                                                 "Определение масштаба изображения позволяет корректно воссоздать размеры объектов на планировке в соответствии с реальными значениями. Необходимо выделить объект, у которого указаны размеры и ввести их в соотвествующие поля."))
        self.sizes_label.setText(_translate("MainWindow", "Размеры выделенной области"))
        self.horizontal_size_label.setText(_translate("MainWindow", "Горизонтальный размер"))
        self.horizontal_size_label_2.setText(_translate("MainWindow", "(мм)"))
        self.vertical_size_label.setText(_translate("MainWindow", "Вертикальный размер"))
        self.mm_vertical_label.setText(_translate("MainWindow", "(мм)"))
        self.calculate_size_push_button.setText(_translate("MainWindow", "Расчет масштаба"))
        self.show_sizes_check_box.setText(_translate("MainWindow", "Отображать размеры выделенной области"))
        self.apply_size_push_button.setText(_translate("MainWindow", "Подтвердить размеры"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.corners_tab), _translate("MainWindow", "Углы и стены"))
        self.corners_detection_information_text.setText(_translate("MainWindow",
                                                                   "Перед обработкой изображения алгоритмом необходимо применить к изображению определенные фильтры, чтобы определение углов происходило более точно. Изображение переводится в черно-белый формат. Threshold - отфильтровывает промежуточные оттенки серого. Erode - расширяет белые области. Dilate - расширяет черные области. После нахождения углов алгоритмом можно подтвердить найденные углы и добавить вручную те, что алгоритм не смог найти. Точки, отвечающие за оконные и дверные проемы необходимо исключить."))
        self.threshold_label.setText(_translate("MainWindow", "Threshold"))
        self.threshold_maxval_label_1.setText(_translate("MainWindow", "Пороговое значение цвета: 0"))
        self.threshold_maxval_label_2.setText(_translate("MainWindow", "255"))
        self.erode_1_label.setText(_translate("MainWindow", "Erode 1"))
        self.erode_1_check_box.setText(_translate("MainWindow", "Использовать erode 1"))
        self.kernel_size_label_2.setText(_translate("MainWindow", "Размер ядра"))
        self.dilate_1_label.setText(_translate("MainWindow", "Dilate 1"))
        self.dilate_1_check_box.setText(_translate("MainWindow", "Использовать dilate 1"))
        self.kernel_size_label_1.setText(_translate("MainWindow", "Размер ядра"))
        self.erode_2_label.setText(_translate("MainWindow", "Erode 2"))
        self.erode_2_check_box.setText(_translate("MainWindow", "Использовать erode 2"))
        self.kernel_size_label_4.setText(_translate("MainWindow", "Размер ядра"))
        self.dilate_2_label.setText(_translate("MainWindow", "Dilate 2"))
        self.dilate_2_check_box.setText(_translate("MainWindow", "Использовать dilate 2"))
        self.kernel_size_label_3.setText(_translate("MainWindow", "Размер ядра"))
        self.show_filters_check_box.setText(_translate("MainWindow", "Отображать фильтры"))
        self.label.setText(_translate("MainWindow", "Поиск углов"))
        self.max_corners_label.setText(_translate("MainWindow", "Максимальное количество углов (с запасом)"))
        self.quality_level_label.setText(_translate("MainWindow", "Качество искомых углов (0 - 1)"))
        self.min_distance_label.setText(_translate("MainWindow", "Минимальное расстояние между углами (в пикселях)"))
        self.find_corners_push_button.setText(_translate("MainWindow", "Обнаружить углы"))
        self.filter_found_corners_label.setText(_translate("MainWindow", "Фильтрация найденных углов"))
        self.delete_corner_radio_button.setText(_translate("MainWindow", "Удалить угол"))
        self.add_corner_radio_button.setText(_translate("MainWindow", "Добавить угол"))
        self.confirm_corner_radio_button.setText(_translate("MainWindow", "Подтвердить угол"))
        self.confirm_corners_push_button.setText(_translate("MainWindow", "Подтвердить углы"))
        self.wall_confirmation_text.setText(_translate("MainWindow",
                                                       "Далее путем последовательного выделения точек необходимо добавить прямые, являющиеся стенами. Необходимо соединить все подтвержденные углы."))
        self.add_wall_radio_button.setText(_translate("MainWindow", "Добавить стену"))
        self.delete_wall_radio_button.setText(_translate("MainWindow", "Удалить стену"))
        self.confirm_walls_push_button.setText(_translate("MainWindow", "Подтвердить стены"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.furniture_tab), _translate("MainWindow", "Элементы интерьера"))
        self.corners_detection_information_text_2.setText(_translate("MainWindow",
                                                                     "Определение объектов интерьера происходит путем выделения одного и поиска на изображении других вариаций расположения и вращения этого элемента, это может быть окно, дверь или мебель. Сначала объект выделяется в прямоугольник, ограничивающий его, или регион интереса. Размер и вращение региона можно настроить. Далее на изображении ищутся такие же объекты и сохраняются для дальнейшего импорта в 3d пакет."))
        self.roi_label.setText(_translate("MainWindow", "Регион интереса"))
        self.roi_rotation_label_1.setText(_translate("MainWindow", "Угол поворота"))
        self.roi_rotation_label_2.setText(_translate("MainWindow", "0"))
        self.roi_rotation_label_3.setText(_translate("MainWindow", "359"))
        self.roi_width_control_label.setText(_translate("MainWindow", "Ширина"))
        self.decrease_roi_width_push_button.setText(_translate("MainWindow", "-"))
        self.increase_roi_width_push_button.setText(_translate("MainWindow", "+"))
        self.add_region_of_interest_radio_button.setText(_translate("MainWindow", "Добавить регион интереса"))
        self.roi_height_control_label.setText(_translate("MainWindow", "Глубина"))
        self.decrease_roi_height_push_button.setText(_translate("MainWindow", "-"))
        self.increase_roi_height_push_button.setText(_translate("MainWindow", "+"))
        self.windows_label.setText(_translate("MainWindow", "Окна"))
        self.window_casements_label.setText(_translate("MainWindow", "Количество створок"))
        self.windows_to_find_label.setText(_translate("MainWindow", "Количество окон для нахождения"))
        self.find_windows_push_button.setText(_translate("MainWindow", "Найти окна"))
        self.confirm_window_radio_button.setText(_translate("MainWindow", "Подтвердить окно"))
        self.add_window_radio_button.setText(_translate("MainWindow", "Добавить окно"))
        self.delete_window_radio_button.setText(_translate("MainWindow", "Удалить окно"))
        self.confirm_windows_push_button.setText(_translate("MainWindow", "Подтвердить окна"))
        self.show_confired_windows_ckeck_box.setText(_translate("MainWindow", "Отображать подтвержденные окна"))
        self.doors_label.setText(_translate("MainWindow", "Двери"))
        self.door_type_label.setText(_translate("MainWindow", "Тип двери"))
        self.door_type_combo_box.setItemText(0, _translate("MainWindow", "Входная дверь"))
        self.door_type_combo_box.setItemText(1, _translate("MainWindow", "Межкомнатная одинарная дверь"))
        self.door_type_combo_box.setItemText(2, _translate("MainWindow", "Межкомнатная двойная дверь"))
        self.doors_to_find_label.setText(_translate("MainWindow", "Количество дверей для нахождения"))
        self.find_doors_push_button.setText(_translate("MainWindow", "Найти двери"))
        self.confirm_door_radio_button.setText(_translate("MainWindow", "Подтвердить дверь"))
        self.add_door_radio_button.setText(_translate("MainWindow", "Добавить дверь"))
        self.delete_door_radio_button.setText(_translate("MainWindow", "Удалить дверь"))
        self.confirm_doors_push_button.setText(_translate("MainWindow", "Подтвердить двери"))
        self.show_confired_doors_ckeck_box.setText(_translate("MainWindow", "Отображать подтвержденные двери"))
        self.furniture_label.setText(_translate("MainWindow", "Мебель"))
        self.furniture_type_label.setText(_translate("MainWindow", "Тип мебели"))

        for i, furniture in enumerate(self.furniture_list):
            self.furniture_type_combo_box.setItemText(i, _translate("MainWindow", furniture['Отображаемое в приложении название']))

        self.furniture_to_find_label.setText(_translate("MainWindow", "Количество объектов для нахождения"))
        self.find_furniture_push_button.setText(_translate("MainWindow", "Найти выделенную мебель"))
        self.confirm_furniture_radio_button.setText(_translate("MainWindow", "Подтвердить мебель"))
        self.add_furniture_radio_button.setText(_translate("MainWindow", "Добавить мебель"))
        self.delete_furniture_radio_button.setText(_translate("MainWindow", "Удалить мебель"))
        self.confirm_furniture_push_button.setText(_translate("MainWindow", "Подтвердить выделенную мебель"))

    # Connecting signals with methods
    def setup_connection(self):
        # Top bar
        self.open_image_action.triggered.connect(self.import_image)
        self.save_data_action.triggered.connect(self.save_data)

        # Tab menu
        self.tabWidget.currentChanged.connect(self.change_tab)

        # Scale tab
        self.calculate_size_push_button.clicked.connect(self.calculate_image_scale)
        self.show_sizes_check_box.stateChanged.connect(self.show_real_sizes)
        self.apply_size_push_button.clicked.connect(self.apply_real_dimensions)

        # Corners tab
        self.threshold_maxval.valueChanged.connect(self.corners_filters)

        self.dilate_1_check_box.stateChanged.connect(self.corners_filters)
        self.dilate_1_spin_box.valueChanged.connect(self.corners_filters)
        self.dilate_1_spin_box.valueChanged.connect(lambda: self.dilate_1_check_box.setChecked(True))

        self.erode_1_check_box.stateChanged.connect(self.corners_filters)
        self.erode_1_spin_box.valueChanged.connect(self.corners_filters)
        self.erode_1_spin_box.valueChanged.connect(lambda: self.erode_1_check_box.setChecked(True))

        self.dilate_2_check_box.stateChanged.connect(self.corners_filters)
        self.dilate_2_spin_box.valueChanged.connect(self.corners_filters)
        self.dilate_2_spin_box.valueChanged.connect(lambda: self.dilate_2_check_box.setChecked(True))

        self.erode_2_check_box.stateChanged.connect(self.corners_filters)
        self.erode_2_spin_box.valueChanged.connect(self.corners_filters)
        self.erode_2_spin_box.valueChanged.connect(lambda: self.erode_2_check_box.setChecked(True))

        self.show_filters_check_box.stateChanged.connect(self.set_image)

        self.find_corners_push_button.clicked.connect(self.find_corners)

        self.add_corner_radio_button.toggled.connect(self.set_corners_found)
        self.confirm_corners_push_button.clicked.connect(self.confirm_corners)

        self.confirm_walls_push_button.clicked.connect(self.apply_corners_and_walls)

        # Interior elements tab
        self.roi_rotation_slider.valueChanged.connect(self.rotate_rectangle)
        self.roi_rotation_slider.installEventFilter(MainWindow)
        self.increase_roi_width_push_button.clicked.connect(self.increase_roi_width)
        self.decrease_roi_width_push_button.clicked.connect(self.decrease_roi_width)
        self.increase_roi_height_push_button.clicked.connect(self.increase_roi_height)
        self.decrease_roi_height_push_button.clicked.connect(self.decrease_roi_height)

        self.find_windows_push_button.clicked.connect(lambda: self.find_furniture('window'))
        self.confirm_windows_push_button.clicked.connect(self.confirm_windows)
        self.show_confired_windows_ckeck_box.stateChanged.connect(self.draw_all_rois)
        self.find_doors_push_button.clicked.connect(lambda: self.find_furniture('door'))
        self.confirm_doors_push_button.clicked.connect(self.confirm_doors)
        self.show_confired_doors_ckeck_box.stateChanged.connect(self.draw_all_rois)
        self.find_furniture_push_button.clicked.connect(lambda:
                    self.find_furniture(self.furniture_list[self.furniture_type_combo_box.currentIndex()]['Тип мебели']))
        self.confirm_furniture_push_button.clicked.connect(self.confirm_furniture)

        # Scale slider
        self.scale_slider.valueChanged.connect(self.update_image_scale)

        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event

    def import_image(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_path, _ = QFileDialog.getOpenFileName(MainWindow, "Import Image", "",
                                                             "Image Files (*.png *.jpg *.bmp)", options=options)

        # Check if a file was selected
        if file_path:
            self.reinitialize_data()
            self.image_path = file_path
            self.initial_image = cv2.imread(self.image_path)
            self.image = cv2.resize(self.initial_image, None, fx=self.IMAGE_UPSCALE_RATE, fy=self.IMAGE_UPSCALE_RATE)
            self.image_with_interior_elements = self.image.copy()
            self.corners_filters()
            self.show_filters_check_box.setChecked(False)
            self.show_sizes_check_box.setChecked(False)
            self.tabWidget.tabBar().setTabTextColor(0, QtCore.Qt.red)
            self.tabWidget.tabBar().setTabTextColor(1, QtCore.Qt.red)
            self.tabWidget.tabBar().setTabTextColor(2, QtCore.Qt.red)
            self.set_image()

    def save_data(self):
        self.adjust_walls_data()
        if self.image.shape[0] > self.image.shape[1]:
            fix_ratio = self.image.shape[1] / self.image.shape[0]
            vertical_image_size = self.image.shape[0] * self.vertical_image_size_scale / 1000
            horizontal_image_size = self.image.shape[1] * self.horizontal_image_size_scale / 1000 / fix_ratio
        else:
            fix_ratio = self.image.shape[0] / self.image.shape[1]
            vertical_image_size = self.image.shape[0] * self.vertical_image_size_scale / 1000 / fix_ratio
            horizontal_image_size = self.image.shape[1] * self.horizontal_image_size_scale / 1000

        converted_windows = []
        if self.confirmed_windows:
            for i, window in enumerate(self.confirmed_windows):
                converted_window = {
                    f'{i + 1}_center': tuple(self.adjust_coordinate(window['center'])),
                    f'{i + 1}_size': tuple(self.adjust_furniture_size(window['size'])),
                    f'{i + 1}_rotation_angle': int(window['rotation_angle']),
                    f'{i + 1}_furniture_name': window['furniture_name'],
                    f'{i + 1}_casements': window['casements']
                }
                converted_windows.append(converted_window)

        converted_doors = []
        if self.confirmed_doors:
            for i, door in enumerate(self.confirmed_doors):
                converted_door = {
                    f'{i + 1}_center': tuple(self.adjust_coordinate(door['center'])),
                    f'{i + 1}_size': tuple(self.adjust_furniture_size(door['size'])),
                    f'{i + 1}_rotation_angle': int(door['rotation_angle']),
                    f'{i + 1}_furniture_name': door['furniture_name'],
                    f'{i + 1}_door_type': door['door_type']
                }
                converted_doors.append(converted_door)

        converted_furniture = []
        if self.confirmed_furniture:
            for i, interior_element in enumerate(self.confirmed_furniture):
                converted_interior_element = {
                    f'{i + 1}_center': tuple(self.adjust_coordinate(interior_element['center'])),
                    f'{i + 1}_size': tuple(self.adjust_furniture_size(interior_element['size'])),
                    f'{i + 1}_rotation_angle': int(interior_element['rotation_angle']),
                    f'{i + 1}_furniture_name': interior_element['furniture_name']
                }
                converted_furniture.append(converted_interior_element)


        data = {
            'image_location': os.path.abspath(self.image_path),
            'horizontal_image_size': horizontal_image_size,
            'vertical_image_size': vertical_image_size,
            'walls': self.adjusted_walls,
            'outside_walls': self.find_outside_walls(),
            'windows': converted_windows,
            'doors': converted_doors,
            'furniture': converted_furniture
        }

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(MainWindow, "Save JSON File", "",
                                                   "JSON Files (*.json);;All Files (*)", options=options)

        if file_path:
            if not file_path.endswith(".json"):
                file_path += ".json"

            with open(file_path, "w") as file:
                json.dump(data, file)

    def reinitialize_data(self):
        self.image_path = None
        self.image = None
        self.initial_image = None

        # Variables for drawing
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.colors = [self.GREEN, self.YELLOW, self.RED]

        self.image_label.setMouseTracking(True)
        self.middle_button_pressed = False
        self.middle_button_start_pos = QtCore.QPoint()
        self.middle_button_start_hscroll_pos = 0
        self.middle_button_start_vscroll_pos = 0

        self.image_with_corners = None
        self.filtered_image = None
        self.filtered_image_with_corners = None
        self.image_with_rectangle = None
        self.corners_found = False
        self.corners = []
        self.confirmed_corners = []
        self.corners_confirmed = False
        self.first_selected_corner = None
        self.walls = []
        self.threshold_current_value = 0
        self.dilate_1 = False
        self.erode_1 = False
        self.scale_factor = 1
        self.horizontal_image_size_scale = 1.0
        self.vertical_image_size_scale = 1.0
        self.real_sizes_are_shown = False
        self.image_real_sizes_scale_calculated = False
        self.real_sizes_confirmed = False
        self.scale_changed = None

        # Variables for drawing rectangles
        self.drawing = False
        self.rectangle = None
        self.rectangle_drawn = False
        self.start_x, self.start_y = -1, -1
        self.current_x, self.current_y = -1, -1

        # Variables for moving rectangle
        self.top_left_delta_x = -1
        self.top_left_delta_y = -1
        self.bottom_right_delta_x = -1
        self.bottom_right_delta_y = -1
        self.roi_center_delta = [-1, -1]

        # Variables for resizing rectangle
        self.start_pos = None
        self.end_pos = None
        self.resizing_rectangle = False
        self.moving_roi = False
        self.resizing_corner = None

        # Variables for interior elements search
        self.image_with_interior_elements = None
        self.region_of_interest = None
        self.found_windows = None
        self.confirmed_windows = None
        self.found_doors = None
        self.confirmed_doors = None
        self.found_furniture = None
        self.confirmed_furniture = None
        self.draw_region_of_interest = False

        # Exporting data
        self.adjusted_walls = []
        self.json_data = None

    def change_tab(self):
        self.set_image()
        if self.tabWidget.currentIndex() != 1:
            self.image_label.setMouseTracking(True)
        else:
            self.image_label.setMouseTracking(False)
        self.draw_rectangles()

    def set_image(self):
        if self.image is not None:
            if self.tabWidget.currentIndex() == 0:
                if self.rectangle_drawn:
                    frame = cv2.cvtColor(self.image_with_rectangle, cv2.COLOR_BGR2RGB)
                else:
                    frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            if self.tabWidget.currentIndex() == 1:
                if self.show_filters_check_box.isChecked():
                    if self.corners_found:
                        frame = cv2.cvtColor(self.filtered_image_with_corners, cv2.COLOR_BGR2RGB)
                    else:
                        if self.filtered_image is None:
                            self.corners_filters()
                        frame = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2RGB)
                else:
                    if self.corners_found:
                        frame = cv2.cvtColor(self.image_with_corners, cv2.COLOR_BGR2RGB)
                    else:
                        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            if self.tabWidget.currentIndex() == 2:
                frame = cv2.cvtColor(self.image_with_interior_elements, cv2.COLOR_BGR2RGB)

            image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))

            self.update_image_scale()

    def update_image_scale(self):
        if self.image is not None:
            old_scale_factor = self.scale_factor
            self.scale_factor = self.scale_slider.value() / 200
            self.image_label.setScaledContents(True)

            # Store the current scroll bar values
            # scroll_pos_h = self.scrollArea.horizontalScrollBar().value()
            # scroll_pos_v = self.scrollArea.verticalScrollBar().value()

            scaled_width = int(self.scale_factor * self.image_label.pixmap().width())
            scaled_height = int(self.scale_factor * self.image_label.pixmap().height())
            self.image_label.setFixedSize(QtCore.QSize(scaled_width, scaled_height))
            if old_scale_factor != self.scale_factor:
                self.scale_changed = True

                # # Calculate the new scroll bar positions based on scale and image size
                # scroll_max_h = self.scrollArea.horizontalScrollBar().maximum()
                # scroll_max_v = self.scrollArea.verticalScrollBar().maximum()
                # new_scroll_pos_h = (scroll_pos_h + (scroll_max_h / 2)) * (self.scale_factor / old_scale_factor) - (
                #             scroll_max_h / 2)
                # new_scroll_pos_v = (scroll_pos_v + (scroll_max_v / 2)) * (self.scale_factor / old_scale_factor) - (
                #             scroll_max_v / 2)
                #
                # # Set the scroll bar values to the middle position
                # self.scrollArea.horizontalScrollBar().setValue(int(new_scroll_pos_h))
                # self.scrollArea.verticalScrollBar().setValue(int(new_scroll_pos_v))

            if self.tabWidget.currentIndex() == 0 and self.image_real_sizes_scale_calculated\
                    and self.real_sizes_are_shown and self.scale_changed:
                self.scale_changed = False
                self.update_rectangle_and_text()

            # Restore the scroll bar values
            # self.scrollArea.horizontalScrollBar().setValue(scroll_pos_h)
            # self.scrollArea.verticalScrollBar().setValue(scroll_pos_v)

    def update_rectangle_and_text(self):
        self.draw_rectangles()
        self.draw_real_sizes_over_rectangle()
        self.set_image()
        self.scale_changed = False

    def mouse_press_event(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self.start_x, self.start_y = self.get_scaled_coordinates(event.x(), event.y())
            # Processing mouse press events in scale tab
            if self.tabWidget.currentIndex() == 0:
                if self.rectangle is not None and self.get_resizing_corner(self.rectangle, self.start_x, self.start_y) is not None:
                    self.resizing_rectangle = True
                    self.top_left_delta_x = self.start_x - self.rectangle[0][0]
                    self.top_left_delta_y = self.start_y - self.rectangle[0][1]
                    self.bottom_right_delta_x = self.start_x - self.rectangle[1][0]
                    self.bottom_right_delta_y = self.start_y - self.rectangle[1][1]

                    self.resizing_corner = self.get_resizing_corner(self.rectangle, self.start_x, self.start_y)
                elif not self.image_real_sizes_scale_calculated:
                    self.drawing = True

            # Processing mouse press events in corners tab
            elif self.tabWidget.currentIndex() == 1:
                if self.corners_found:
                    selected_corner = self.find_selected_corner(self.start_x, self.start_y)
                    # Confirm corner processing
                    if self.confirm_corner_radio_button.isChecked():
                        if selected_corner is not None:
                            if selected_corner not in self.confirmed_corners:
                                self.confirmed_corners.append(selected_corner)
                    # Add corner processing
                    elif self.add_corner_radio_button.isChecked():
                        if [self.start_x, self.start_y] not in self.confirmed_corners:
                            self.confirmed_corners.append([self.start_x, self.start_y])
                    # Delete corner processing
                    elif self.delete_corner_radio_button.isChecked():
                        if selected_corner in self.confirmed_corners:
                            self.confirmed_corners.remove(selected_corner)
                        if selected_corner in self.corners:
                            self.corners.remove(selected_corner)
                        elif selected_corner in self.corners:
                            self.corners.remove(selected_corner)
                    # Add wall processing
                    elif self.add_wall_radio_button.isChecked():
                        if selected_corner in self.confirmed_corners:
                            if self.first_selected_corner is None:
                                self.first_selected_corner = selected_corner
                            else:
                                if not is_corner_in_walls(self.walls, selected_corner):
                                    self.walls.append([self.first_selected_corner, selected_corner])
                                    self.first_selected_corner = selected_corner
                                else:
                                    self.walls.append([self.first_selected_corner, selected_corner])
                                    self.first_selected_corner = None
                        else:
                            self.first_selected_corner = None
                    # Delete wall processing
                    elif self.delete_wall_radio_button.isChecked():
                        if selected_corner in self.confirmed_corners:
                            if self.first_selected_corner is None:
                                self.first_selected_corner = selected_corner
                            else:
                                if self.check_and_remove_wall(self.walls, self.first_selected_corner, selected_corner):
                                    self.first_selected_corner = None
                                elif self.check_and_remove_wall(self.outside_walls, self.first_selected_corner, selected_corner):
                                    self.first_selected_corner = None
                        else:
                            self.first_selected_corner = None

                    self.image_with_corners = self.image.copy()
                    self.filtered_image_with_corners = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2BGR)
                    if not self.corners_confirmed:
                        self.draw_corner_marks(self.corners, self.RED)
                    self.draw_corner_marks(self.confirmed_corners, self.GREEN)

            # Processing mouse press events in interior elements tab
            elif self.tabWidget.currentIndex() == 2:
                if self.draw_region_of_interest \
                        and self.region_of_interest.is_point_inside([self.start_x, self.start_y]):
                    self.moving_roi = True
                    self.roi_center_delta = [self.region_of_interest.center[0] - self.start_x,
                                               self.region_of_interest.center[1] - self.start_y]

                elif self.add_region_of_interest_radio_button.isChecked():
                    self.drawing = True

                else:

                    # Windows
                    selected_window_region_in_found = is_point_in_found_elements(self.found_windows,
                                                                        [self.start_x, self.start_y])
                    selected_window_region_in_confirmed = is_point_in_found_elements(self.confirmed_windows,
                                                                        [self.start_x, self.start_y])
                    if self.confirm_window_radio_button.isChecked() and selected_window_region_in_found is not None:
                        if self.confirmed_windows is None:
                            self.confirmed_windows = [selected_window_region_in_found]
                        else:
                            self.confirmed_windows.append(selected_window_region_in_found)
                        self.show_confired_windows_ckeck_box.setChecked(True)
                    elif self.add_window_radio_button.isChecked():
                        self.draw_region_of_interest = True
                        self.region_of_interest.center = (self.start_x, self.start_y)
                    elif self.delete_window_radio_button.isChecked():
                        selected_window_region_in_found is not None\
                                        and self.found_windows.remove(selected_window_region_in_found)
                        selected_window_region_in_confirmed is not None\
                                        and self.confirmed_windows.remove(selected_window_region_in_confirmed)

                    # Doors
                    selected_door_region_in_found = is_point_in_found_elements(self.found_doors,
                                                                        [self.start_x, self.start_y])
                    selected_door_region_in_confirmed = is_point_in_found_elements(self.confirmed_doors,
                                                                        [self.start_x, self.start_y])
                    if self.confirm_door_radio_button.isChecked() and selected_door_region_in_found is not None:
                        if self.confirmed_doors is None:
                            self.confirmed_doors = [selected_door_region_in_found]
                        else:
                            self.confirmed_doors.append(selected_door_region_in_found)
                        self.show_confired_doors_ckeck_box.setChecked(True)
                    elif self.add_door_radio_button.isChecked():
                        self.draw_region_of_interest = True
                        self.region_of_interest.center = (self.start_x, self.start_y)
                    elif self.delete_door_radio_button.isChecked():
                        selected_door_region_in_found is not None\
                                        and self.found_doors.remove(selected_door_region_in_found)
                        selected_door_region_in_confirmed is not None\
                                        and self.confirmed_doors.remove(selected_door_region_in_confirmed)

                    # Furniture
                    selected_furniture_region_in_found = is_point_in_found_elements(
                        self.found_furniture, [self.start_x, self.start_y])
                    selected_furniture_region_in_confirmed = is_point_in_found_elements(
                        self.confirmed_furniture, [self.start_x, self.start_y])
                    if self.confirm_furniture_radio_button.isChecked() and selected_furniture_region_in_found is not None:
                        if self.confirmed_furniture is None:
                            self.confirmed_furniture = [selected_furniture_region_in_found]
                        else:
                            self.confirmed_furniture.append(selected_furniture_region_in_found)
                    elif self.add_furniture_radio_button.isChecked():
                        self.draw_region_of_interest = True
                        self.region_of_interest.center = (self.start_x, self.start_y)
                    elif self.delete_furniture_radio_button.isChecked():
                        if selected_furniture_region_in_found is not None:
                            self.found_furniture.remove(selected_furniture_region_in_found)
                        if selected_furniture_region_in_confirmed is not None:
                            self.confirmed_furniture.remove(selected_furniture_region_in_confirmed)

                self.draw_all_rois(mouse_move=False)

        # Processing middle mouse button navigation
        elif event.button() == QtCore.Qt.MiddleButton:
            self.middle_button_pressed = True
            self.middle_button_start_pos = event.pos()
            self.middle_button_start_hscroll_pos = self.scrollArea.horizontalScrollBar().value()
            self.middle_button_start_vscroll_pos = self.scrollArea.verticalScrollBar().value()

    def mouse_move_event(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            if self.tabWidget.currentIndex() != 1:
                if self.resizing_rectangle:
                    self.resize_rectangle(event.pos())
                    if self.real_sizes_are_shown:
                        self.draw_real_sizes_over_rectangle()
                elif self.drawing:
                    self.current_x, self.current_y = self.get_scaled_coordinates(event.x(), event.y())
                    if self.tabWidget.currentIndex() == 0:
                        self.draw_rectangles()
                    else:
                        self.draw_all_rois()
                elif self.moving_roi:
                    self.move_roi(event.pos())
                    self.draw_all_rois()
        elif self.middle_button_pressed:
            delta = event.pos() - self.middle_button_start_pos
            new_hscroll_pos = self.middle_button_start_hscroll_pos - delta.x()
            new_vscroll_pos = self.middle_button_start_vscroll_pos - delta.y()
            self.scrollArea.horizontalScrollBar().setValue(new_hscroll_pos)
            self.scrollArea.verticalScrollBar().setValue(new_vscroll_pos)

        cursor_shape = QtCore.Qt.ArrowCursor
        if self.tabWidget.currentIndex() == 0:
            if self.rectangle:
                x, y = self.get_scaled_coordinates(event.x(), event.y())
                # print(f'scaled cursor:{(x, y)}, cursor: {(event.x(), event.y())},'
                #       f' top left scaled corner: {self.get_scaled_coordinates(*self.rectangle[0])}, top left corner: {self.rectangle[0]}')
                over_position = self.get_resizing_corner(self.rectangle, x, y)
                if over_position == "top_left" or over_position == "bottom_right":
                    cursor_shape = QtCore.Qt.SizeFDiagCursor
                elif over_position == "top_right" or over_position == "bottom_left":
                    cursor_shape = QtCore.Qt.SizeBDiagCursor
                elif over_position == "top_edge" or over_position == "bottom_edge":
                    cursor_shape = QtCore.Qt.SizeVerCursor
                elif over_position == "left_edge" or over_position == "right_edge":
                    cursor_shape = QtCore.Qt.SizeHorCursor
                elif over_position == 'inside':
                    cursor_shape = QtCore.Qt.ClosedHandCursor
        if self.tabWidget.currentIndex() == 2:
            if self.region_of_interest and self.draw_region_of_interest:
                if self.region_of_interest.is_point_inside(self.get_scaled_coordinates(event.x(), event.y())):
                    cursor_shape = QtCore.Qt.ClosedHandCursor

        self.image_label.setCursor(cursor_shape)

    def mouse_release_event(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.tabWidget.currentIndex() != 1:
                if self.resizing_rectangle and self.resizing_corner:
                    self.resizing_rectangle = False
                    self.resizing_corner = None
                    self.draw_rectangles()
                    if self.real_sizes_are_shown:
                        self.draw_real_sizes_over_rectangle()
                elif self.moving_roi:
                    self.moving_roi = False
                    self.draw_rectangles()
                elif self.drawing:
                    self.drawing = False
                    self.current_x, self.current_y = self.get_scaled_coordinates(event.x(), event.y())
                    self.draw_rectangles(save=True)
        elif event.button() == QtCore.Qt.MiddleButton:
            self.middle_button_pressed = False

    def calculate_image_scale(self):
        if self.rectangle is not None:
            self.horizontal_image_size_scale = self.horizontal_size_spin_box.value() / abs(self.rectangle[0][0] - self.rectangle[1][0])
            self.vertical_image_size_scale = self.vertical_size_spin_box.value() / abs(self.rectangle[0][1] - self.rectangle[1][1])
            self.show_sizes_check_box.setChecked(True)
            self.image_real_sizes_scale_calculated = True
            if self.real_sizes_are_shown:
                self.update_rectangle_and_text()
        else:
            self.no_size_rectangle_drawn_box.exec_()

    def corners_filters(self):
        self.corners_found = False
        self.show_filters_check_box.setChecked(True)
        kernel_erode_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode_1_spin_box.value(), self.erode_1_spin_box.value()))
        kernel_dilate_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate_1_spin_box.value(), self.dilate_1_spin_box.value()))
        kernel_erode_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode_2_spin_box.value(), self.erode_2_spin_box.value()))
        kernel_dilate_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate_2_spin_box.value(), self.dilate_2_spin_box.value()))

        if self.image is not None:
            filtered_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, filtered_image = cv2.threshold(filtered_image, self.threshold_maxval.value(), 255, cv2.THRESH_BINARY)

            if self.erode_1_check_box.isChecked():
                filtered_image = cv2.dilate(filtered_image, kernel_erode_1, iterations=1)
            if self.dilate_1_check_box.isChecked():
                filtered_image = cv2.erode(filtered_image, kernel_dilate_1, iterations=1)
            if self.erode_2_check_box.isChecked():
                filtered_image = cv2.dilate(filtered_image, kernel_erode_2, iterations=1)
            if self.dilate_2_check_box.isChecked():
                filtered_image = cv2.erode(filtered_image, kernel_dilate_2, iterations=1)

            self.filtered_image = filtered_image
            self.set_image()


    # Finding corners with cornerHarris
    def find_corners(self):

        # Apply cornerHarris algorithm
        dst = cv2.cornerHarris(self.filtered_image, blockSize=2 * self.IMAGE_UPSCALE_RATE,
                               ksize=3 * self.IMAGE_UPSCALE_RATE, k=0.04)

        # Define a threshold to identify strong corners
        threshold = self.quality_level_spin_box.value() * dst.max()

        # Find corner coordinates
        corners = np.argwhere(dst > threshold)
        corners = corners.tolist()

        if corners:
            for i in range(len(corners)):
                corners[i] = [corners[i][1], corners[i][0]]

            # Refine corner locations using cornerSubPix
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners = np.float32(corners)
            cv2.cornerSubPix(self.filtered_image, corners, (5, 5), (-1, -1), criteria)
            corners = np.intp(corners)

            self.image_with_corners = self.image.copy()
            self.filtered_image_with_corners = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2BGR)
            self.corners_found = True
            self.corners = corners.tolist()
            self.draw_corner_marks(self.corners, self.RED)

        if self.confirmed_corners is not None:
            self.draw_corner_marks(self.confirmed_corners, self.GREEN)

        self.set_image()

    def find_outside_walls(self):
        outside_corners = cv2.convexHull(np.array(self.adjust_coordinates(self.confirmed_corners), dtype=np.float32))
        outside_corners_new = []
        for corner in outside_corners:
            x = corner[0][0]
            y = corner[0][1]
            corner_new = (x, y)
            outside_corners_new.append(corner_new)
        outside_walls = []
        EPSILON = 1e-5

        for wall in self.adjusted_walls:
            start_point, end_point = wall
            if any(np.allclose(start_point, corner, atol=EPSILON) for corner in outside_corners_new) and \
                    any(np.allclose(end_point, corner, atol=EPSILON) for corner in outside_corners_new):
                outside_walls.append(wall)
            elif any(np.allclose(point, corner, atol=EPSILON) for point in (start_point, end_point) for corner in
                     outside_corners_new):
                outside_walls.append(wall)
        return outside_walls

    def find_furniture(self, furniture_name):
        found_furniture = self.prepare_template_and_find_variations(furniture_name)
        self.found_windows = None
        if len(found_furniture) > 0:
            self.draw_region_of_interest = False
            for furniture_variation in found_furniture:
                furniture_variation['center'] = tuple(value * self.IMAGE_UPSCALE_RATE for value in furniture_variation['center'])
                furniture_variation['size'] = tuple(value * self.IMAGE_UPSCALE_RATE for value in furniture_variation['size'])
                if furniture_name == 'window':
                    furniture_variation['casements'] = self.windows_casements_spin_box.value()
                if furniture_name == 'door':
                    furniture_variation['door_type'] = self.get_door_type()
            if furniture_name == 'window':
                if self.found_windows is not None:
                    self.found_windows.append(found_furniture)
                else:
                    self.found_windows = found_furniture
            elif furniture_name == 'door':
                if self.found_doors is not None:
                    self.found_doors.append(found_furniture)
                else:
                    self.found_doors = found_furniture
            else:
                if self.found_furniture is not None:
                    self.found_furniture.append(found_furniture)
                else:
                    self.found_furniture = found_furniture
            self.rectangle_drawn = False
            self.draw_all_rois()

    def confirm_windows(self):
        self.show_confired_windows_ckeck_box.setChecked(True)
        self.found_windows = None
        if self.draw_region_of_interest:
            self.draw_region_of_interest = False
            variation = {
                'center': tuple(int(value) for value in self.region_of_interest.center),
                'size': self.region_of_interest.size,
                'rotation_angle': self.region_of_interest.angle,
                'furniture_name': 'window',
                'casements': self.windows_casements_spin_box.value()
            }
            if self.confirmed_windows is not None:
                self.confirmed_windows.append(variation)
            else:
                self.confirmed_windows = [variation]
        if self.confirmed_windows and self.confirmed_doors:
            self.tabWidget.tabBar().setTabTextColor(2, QtCore.Qt.black)
        self.draw_all_rois()

    def confirm_doors(self):
        self.show_confired_doors_ckeck_box.setChecked(True)
        self.found_doors = None
        if self.draw_region_of_interest:
            self.draw_region_of_interest = False
            variation = {
                'center': tuple(int(value) for value in self.region_of_interest.center),
                'size': self.region_of_interest.size,
                'rotation_angle': self.region_of_interest.angle,
                'furniture_name': 'door',
                'door_type': self.get_door_type()
            }
            if self.confirmed_doors is not None:
                self.confirmed_doors.append(variation)
            else:
                self.confirmed_doors = [variation]
        if self.confirmed_windows and self.confirmed_doors:
            self.tabWidget.tabBar().setTabTextColor(2, QtCore.Qt.black)
        self.draw_all_rois()

    def confirm_furniture(self):
        furniture_name = self.furniture_list[self.furniture_type_combo_box.currentIndex()]['Тип мебели']
        self.found_furniture = None
        if self.draw_region_of_interest:
            self.draw_region_of_interest = False
            variation = {
                'center': tuple(int(value) for value in self.region_of_interest.center),
                'size': self.region_of_interest.size,
                'rotation_angle': self.region_of_interest.angle,
                'furniture_name': furniture_name
            }
            if self.confirmed_furniture is not None:
                self.confirmed_furniture.append(variation)
            else:
                self.confirmed_furniture = [variation]
        self.draw_all_rois()

    def prepare_template_and_find_variations(self, furniture_name):
        size = tuple(value // self.IMAGE_UPSCALE_RATE for value in self.region_of_interest.size)
        center = tuple(value // self.IMAGE_UPSCALE_RATE for value in self.region_of_interest.center)
        transformation_matrix = cv2.getRotationMatrix2D(center, self.region_of_interest.angle, 1.0)
        rotated_image = cv2.warpAffine(self.initial_image, transformation_matrix, (self.initial_image.shape[1], self.initial_image.shape[0]))
        template = cv2.getRectSubPix(rotated_image, size, center)
        found_variations = find_template_variations(self.initial_image, template, furniture_name, self.windows_to_find_spin_box.value(),
                                                 threshold=0.9)
        return found_variations
    
    def set_corners_found(self):
        self.corners_found = True

    def confirm_corners(self):
        self.image_with_corners = self.image.copy()
        self.filtered_image_with_corners = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2BGR)
        self.corners = []
        self.draw_corner_marks(self.confirmed_corners, self.GREEN)

    def check_and_remove_wall(self, walls, corner_1, corner_2):
        for wall in walls:
            if [corner_1, corner_2] == wall or [corner_2, corner_1] == wall:
                self.walls.remove(wall)
                return True

    def show_real_sizes(self):
        if self.rectangle:
            self.draw_rectangles()
            if self.show_sizes_check_box.isChecked():
                self.real_sizes_are_shown = True
                self.draw_real_sizes_over_rectangle()
            else:
                self.real_sizes_are_shown = False

    def apply_real_dimensions(self):
        if self.image_real_sizes_scale_calculated:
            self.real_sizes_confirmed = True
            self.tabWidget.tabBar().setTabTextColor(0, QtCore.Qt.black)
            self.draw_real_sizes_over_rectangle()
        else:
            print('Показать вспывающее окно о необходимости найти реальные размеры')

    def apply_corners_and_walls(self):
        if len(self.walls) > 0:
            self.confirmed_walls = self.walls.copy()
            # self.walls = []
            self.tabWidget.tabBar().setTabTextColor(1, QtCore.Qt.black)
            self.draw_walls()
            self.draw_corner_marks(self.confirmed_corners, self.GREEN)
            self.set_image()
        else:
            print('показать выплывающее окно о необходимости разметки стен')

    def draw_corner_marks(self, corners, color):
        if len(corners) > 0:
            for corner in corners:
                x, y = corner
                if corner == self.first_selected_corner:
                    cv2.circle(self.image_with_corners, (x, y), 15, self.RED, 2)
                    cv2.circle(self.image_with_corners, (x, y), 4, self.RED, 2)
                    cv2.circle(self.filtered_image_with_corners, (x, y), 15, self.RED, 2)
                    cv2.circle(self.filtered_image_with_corners, (x, y), 4, self.RED, 2)
                else:
                    cv2.circle(self.image_with_corners, (x, y), 15, color, 2)
                    cv2.circle(self.image_with_corners, (x, y), 4, color, 2)
                    cv2.circle(self.filtered_image_with_corners, (x, y), 15, color, 2)
                    cv2.circle(self.filtered_image_with_corners, (x, y), 4, color, 2)
        self.draw_walls()

        self.set_image()

    def draw_walls(self):
        if self.walls:
            for wall in self.walls:
                start_pos, end_pos = wall
                cv2.line(self.image_with_corners, start_pos, end_pos, self.RED, thickness=2)
                cv2.line(self.filtered_image_with_corners, start_pos, end_pos, self.RED, thickness=2)
        if self.confirmed_walls:
            for wall in self.confirmed_walls:
                start_pos, end_pos = wall
                cv2.line(self.image_with_corners, start_pos, end_pos, self.GREEN, thickness=self.IMAGE_UPSCALE_RATE)
                cv2.line(self.filtered_image_with_corners, start_pos, end_pos, self.GREEN, thickness=self.IMAGE_UPSCALE_RATE)

    def draw_rectangles(self, save=False):
        if self.image is not None:
            self.image_with_rectangle = self.image.copy()
            start_point = (min(self.start_x, self.current_x), min(self.start_y, self.current_y))
            end_point = (max(self.start_x, self.current_x), max(self.start_y, self.current_y))

            # Saving drawn rectangle
            if save:
                if self.tabWidget.currentIndex() == 0:
                    self.rectangle = [start_point, end_point]
                else:
                    size = ((end_point[0] - start_point[0]), (end_point[1] - start_point[1]))
                    center = ((end_point[0] + start_point[0])/2, (end_point[1] + start_point[1])/2)
                    self.region_of_interest = RotatedRectangle(center, size, 0)
                    self.draw_region_of_interest = True

            # Drawing rectangles
            if self.tabWidget.currentIndex() == 0:
                if self.resizing_rectangle:
                    cv2.rectangle(self.image_with_rectangle, self.rectangle[0], self.rectangle[1], self.RED, 1)
                elif self.drawing:
                    cv2.rectangle(self.image_with_rectangle, start_point, end_point, self.GREEN, 2)
                elif self.rectangle:
                    cv2.rectangle(self.image_with_rectangle, self.rectangle[0], self.rectangle[1], self.GREEN, 2)

            elif self.tabWidget.currentIndex() == 2:
                if self.drawing:
                    cv2.rectangle(self.image_with_interior_elements, start_point, end_point, self.RED, 2)
                elif self.draw_region_of_interest:
                    self.region_of_interest.draw_rotated_rectangle(self.image_with_interior_elements, color=self.RED)

            self.rectangle_drawn = True
            self.set_image()

    def draw_rois(self, rois, color):
        for roi in rois:
            center = roi['center']
            size = roi['size']
            rotation_angle = int(roi['rotation_angle'])
            furniture_name = roi['furniture_name']

            current_rotated_rectangle = RotatedRectangle(center, size, rotation_angle)
            current_rotated_rectangle.draw_rotated_rectangle(self.image_with_interior_elements, color=color)

            if size[0] > size[1] and 90 < rotation_angle < 180:
                half_size = size[0] // 2
            else:
                half_size = size[1] // 2
            sin_angle = math.sin(math.radians(rotation_angle))
            # cos_angle = math.cos(math.radians(rotation_angle))
            vertical_offset = int(half_size + abs(sin_angle))

            cv2.putText(self.image_with_interior_elements, f"{rotation_angle} degrees", (center[0],
                        (center[1] - vertical_offset)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.IMAGE_UPSCALE_RATE, color, self.IMAGE_UPSCALE_RATE)
            cv2.putText(self.image_with_interior_elements, furniture_name, (center[0],
                        (center[1] - vertical_offset - 40 * self.IMAGE_UPSCALE_RATE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.IMAGE_UPSCALE_RATE, color, self.IMAGE_UPSCALE_RATE)
            cv2.putText(self.image_with_interior_elements, f'Size: {size[0]} x {size[1]}', (center[0],
                        (center[1] - vertical_offset - 20 * self.IMAGE_UPSCALE_RATE)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.IMAGE_UPSCALE_RATE, color, self.IMAGE_UPSCALE_RATE)

    def draw_all_rois(self, mouse_move=True):
        self.image_with_interior_elements = self.image.copy()
        self.found_windows and self.draw_rois(self.found_windows, self.RED)
        if self.show_confired_windows_ckeck_box.isChecked():
            self.confirmed_windows and self.draw_rois(self.confirmed_windows, self.GREEN)
        self.found_doors and self.draw_rois(self.found_doors, self.RED)
        if self.show_confired_doors_ckeck_box.isChecked():
            self.confirmed_doors and self.draw_rois(self.confirmed_doors, self.GREEN)
        self.found_furniture and self.draw_rois(self.found_furniture, self.RED)
        self.confirmed_furniture and self.draw_rois(self.confirmed_furniture, self.GREEN)
        if self.draw_region_of_interest or (self.drawing and mouse_move):
            self.draw_rectangles()
        self.set_image()

    def draw_real_sizes_over_rectangle(self):
        horizontal_size = abs(self.rectangle[1][0] - self.rectangle[0][0]) * self.horizontal_image_size_scale
        vertical_size = abs(self.rectangle[1][1] - self.rectangle[0][1]) * self.vertical_image_size_scale
        color = self.GREEN if self.real_sizes_confirmed else self.RED
        cv2.putText(self.image_with_rectangle, f'{int(horizontal_size)} mm x {int(vertical_size)} mm',
                    (self.rectangle[0][0], self.rectangle[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1 / self.scale_factor,
                    color, thickness=int(2 / self.scale_factor))
        self.set_image()

    def get_resizing_corner(self, rect, x, y):
        tolerance = 20

        top_left = (rect[0])
        top_right = (rect[1][0], rect[0][1])
        bottom_left = (rect[0][0], rect[1][1])
        bottom_right = (rect[1])

        if abs(x - top_left[0]) <= tolerance and abs(y - top_left[1]) <= tolerance:
            return "top_left"
        elif abs(x - top_right[0]) <= tolerance and abs(y - top_right[1]) <= tolerance:
            return "top_right"
        elif abs(x - bottom_left[0]) <= tolerance and abs(y - bottom_left[1]) <= tolerance:
            return "bottom_left"
        elif abs(x - bottom_right[0]) <= tolerance and abs(y - bottom_right[1]) <= tolerance:
            return "bottom_right"
        elif (top_left[0] + tolerance) < x < (top_right[0] - tolerance) and abs(y - top_right[1]) <= tolerance:
            return "top_edge"
        elif (bottom_left[0] + tolerance) < x < (bottom_right[0] - tolerance) and abs(y - bottom_left[1]) <= tolerance:
            return "bottom_edge"
        elif (top_left[1] + tolerance) < y < (bottom_left[1] - tolerance) and abs(x - top_left[0]) <= tolerance:
            return "left_edge"
        elif (top_right[1] + tolerance) < y < (bottom_right[1] - tolerance) and abs(x - top_right[0]) <= tolerance:
            return "right_edge"
        elif top_left[0] < x < top_right[0] and top_left[1] < y < bottom_left[1]:
            return "inside"
        else:
            return None

    def resize_rectangle(self, pos):
        if self.resizing_rectangle and self.resizing_corner:
            x, y = self.get_scaled_coordinates(pos.x(), pos.y())
            match self.resizing_corner:
                case 'top_left':
                    self.rectangle[0] = (x, y)
                case 'top_right':
                    self.rectangle[1] = (x, self.rectangle[1][1])
                    self.rectangle[0] = (self.rectangle[0][0], y)
                case 'bottom_left':
                    self.rectangle[0] = (x, self.rectangle[0][1])
                    self.rectangle[1] = (self.rectangle[1][0], y)
                case 'bottom_right':
                    self.rectangle[1] = (x, y)
                case 'top_edge':
                    self.rectangle[0] = (self.rectangle[0][0], y)
                case 'bottom_edge':
                    self.rectangle[1] = (self.rectangle[1][0], y)
                case 'left_edge':
                    self.rectangle[0] = (x, self.rectangle[0][1])
                case 'right_edge':
                    self.rectangle[1] = (x, self.rectangle[1][1])
                case 'inside':
                    self.rectangle[0] = (x - self.top_left_delta_x, y - self.top_left_delta_y)
                    self.rectangle[1] = (x - self.bottom_right_delta_x, y - self.bottom_right_delta_y)
            self.draw_rectangles()

    def move_roi(self, pos):
        if self.moving_roi:
            x, y = self.get_scaled_coordinates(pos.x(), pos.y())
            delta_x, delta_y = self.roi_center_delta
            self.region_of_interest.center = [x + delta_x, y + delta_y]

    def increase_roi_width(self):
        self.region_of_interest.size = (self.region_of_interest.size[0] + self.IMAGE_UPSCALE_RATE, self.region_of_interest.size[1])
        self.resize_roi()

    def decrease_roi_width(self):
        self.region_of_interest.size = (self.region_of_interest.size[0] - self.IMAGE_UPSCALE_RATE, self.region_of_interest.size[1])
        self.resize_roi()

    def increase_roi_height(self):
        self.region_of_interest.size = (self.region_of_interest.size[0], self.region_of_interest.size[1] + self.IMAGE_UPSCALE_RATE)
        self.resize_roi()

    def decrease_roi_height(self):
        self.region_of_interest.size = (self.region_of_interest.size[0], self.region_of_interest.size[1] - self.IMAGE_UPSCALE_RATE)
        self.resize_roi()

    def resize_roi(self):
        self.draw_all_rois()

    def rotate_rectangle(self, angle):
        if self.region_of_interest is not None:
            self.region_of_interest.angle = angle
        self.roi_rotation_label_1.setText(f'Угол поворота ({angle})')
        self.roi_rotation_angle_label.setText(str(angle))
        self.roi_rotation_angle_label.adjustSize()
        self.draw_all_rois()

    def get_scaled_coordinates(self, x, y):
        scaled_x = int(x / self.scale_factor)
        scaled_y = int(y / self.scale_factor)
        return scaled_x, scaled_y

    def find_selected_corner(self, x, y):
        for corner in self.corners:
            corner_x, corner_y = corner
            if is_coordinate_inside_circle(x, y, corner_x, corner_y, self.SELECTING_TOLERANCE * self.IMAGE_UPSCALE_RATE):
                return corner

        for corner in self.confirmed_corners:
            corner_x, corner_y = corner
            if is_coordinate_inside_circle(x, y, corner_x, corner_y, self.SELECTING_TOLERANCE * self.IMAGE_UPSCALE_RATE):
                return corner

    def adjust_walls_data(self):
        center_x = self.image.shape[1] // 2
        center_y = self.image.shape[0] // 2

        if self.confirmed_walls:
            for wall in self.confirmed_walls:
                # coordinate_horizontal_adjust_ratio = self.horizontal_image_size_scale * self.IMAGE_UPSCALE_RATE
                # coordinate_vertical_adjust_ratio = self.vertical_image_size_scale * self.IMAGE_UPSCALE_RATE

                coordinate_horizontal_adjust_ratio = self.horizontal_image_size_scale
                coordinate_vertical_adjust_ratio = self.vertical_image_size_scale
                adjusted_start = ((wall[0][0] - center_x) * coordinate_horizontal_adjust_ratio / 1000,
                                  -(wall[0][1] - center_y) * coordinate_vertical_adjust_ratio / 1000)
                adjusted_end = ((wall[1][0] - center_x) * coordinate_horizontal_adjust_ratio / 1000,
                                -(wall[1][1] - center_y) * coordinate_vertical_adjust_ratio / 1000)
                adjusted_wall = [adjusted_start, adjusted_end]
                self.adjusted_walls.append(adjusted_wall)
        else:
            print('ai ai no walls')

    def adjust_coordinate(self, coordinate):
        center_x = self.image.shape[1] // 2
        center_y = self.image.shape[0] // 2
        adjusted_coordinate = ((coordinate[0] - center_x) * self.horizontal_image_size_scale / 1000,
                          -(coordinate[1] - center_y) * self.vertical_image_size_scale / 1000)
        return adjusted_coordinate

    def adjust_coordinates(self, coordinates):
        adjusted_coordinates = []
        for coordinate in coordinates:
            adjusted_coordinates.append(self.adjust_coordinate(coordinate))
        return adjusted_coordinates

    def adjust_furniture_size(self, size):
        return (size[0] * self.horizontal_image_size_scale / 1000, size[1] * self.vertical_image_size_scale / 1000)

    def get_door_type(self):
        if self.door_type_combo_box.currentIndex() == 0:
            door_type = 'entrance_door'
        elif self.door_type_combo_box.currentIndex() == 1:
            door_type = 'single_interior_door'
        else:
            door_type = 'double_interior_door'
        return door_type

    # def handle_wheel_event(self, event):
    #     if event.modifiers() == QtCore.Qt.ShiftModifier:
    #         wheel_angle = event.angleDelta()[1] / 120.0
    # 
    #         if wheel_angle > 0 and self.scale_slider.value() < 400:
    #             self.scale_slider.setValue(self.scale_slider.value() + 1)
    #         elif wheel_angle < 0 and self.scale_slider.value() > 50:
    #             self.scale_slider.setValue(self.scale_slider.value() - 1)
    # 
    #         self.update_image_scale()
    #     else:
    #         # Call the base wheel event handler
    #         super().wheelEvent(event)


class RotatedRectangle:
    def __init__(self, center, size, angle):
        self.center = center
        self.size = size
        self.angle = angle

    def draw_rotated_rectangle(self, image, color=(0, 0, 255), thickness=2, arrow_size=0.4):
        rect = cv2.boxPoints(((self.center[0], self.center[1]), (self.size[0], self.size[1]), self.angle))
        rect = np.intp(rect)

        # Calculate the coordinates of the arrow tip
        arrow_tip_x = int(self.center[0] + self.size[1] * 0.5 * np.cos(np.radians(self.angle + 90)))
        arrow_tip_y = int(self.center[1] + self.size[1] * 0.5 * np.sin(np.radians(self.angle + 90)))
        arrow_tip = (arrow_tip_x, arrow_tip_y)
        center = (int(self.center[0]), int(self.center[1]))

        # Draw the arrow line
        cv2.arrowedLine(image, center, arrow_tip, color, thickness, tipLength=arrow_size)

        # Draw the rotated rectangle
        cv2.drawContours(image, [rect], 0, color, thickness)


    def is_point_inside(self, point):
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        cos_angle = math.cos(math.radians(self.angle))
        sin_angle = math.sin(math.radians(self.angle))
        rotated_x = cos_angle * dx + sin_angle * dy
        rotated_y = -sin_angle * dx + cos_angle * dy

        half_width = self.size[0] / 2
        half_height = self.size[1] / 2

        if abs(rotated_x) <= half_width and abs(rotated_y) <= half_height:
            return True
        else:
            return False


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    MainWindow.show()

    # try:
    #     ui.furniture_list = import_xlsx_config()
    # except PermissionError:
    #     QtWidgets.QMessageBox.critical(None, 'Отказано в доступе к файлу конфигурации',
    #                          'Отказано в доступе к файлу конфигурации. Пожалуйста проверьте наличие файла с конфигурацией в папке с приложением. Закройте Excel с конфигурацией, если файл открыт в нем.')
    #     app.quit()

    sys.exit(app.exec_())
