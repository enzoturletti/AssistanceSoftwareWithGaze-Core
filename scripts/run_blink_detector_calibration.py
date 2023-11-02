#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
from tools.blink_detector import BlinkDetector
from tools.facial_landmarks_detector import FaceLandmarksDetector
from tools.frame_getter import FrameGetter
from tools.config_instance import ConfigInstance
import os
from enum import Enum

class States(str, Enum):
    OPEN_EYE_1 = 1
    OPEN_EYE_2 = 2
    OPEN_EYE_3 = 3
    OPEN_EYE_4 = 4
    OPEN_EYE_5 = 5
    CLOSE_EYE = 6
    METRIC = 7

def main():
    state = States.OPEN_EYE_1

    config_intance = ConfigInstance()

    output_folder = config_intance.click_calibration_path
    output = os.path.join(output_folder,"click_calibrations_results.txt")
    reference_open_eye_1 = config_intance.reference_open_eye_1
    reference_open_eye_2 = config_intance.reference_open_eye_2
    reference_open_eye_3 = config_intance.reference_open_eye_3
    reference_open_eye_4 = config_intance.reference_open_eye_4
    reference_open_eye_5 = config_intance.reference_open_eye_5
    reference_close_eye =  config_intance.reference_close_eye

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if os.path.isfile(output):
        os.remove(output)

    if os.path.isfile(reference_open_eye_1):
        os.remove(reference_open_eye_1)

    if os.path.isfile(reference_open_eye_2):
        os.remove(reference_open_eye_2)
        
    if os.path.isfile(reference_open_eye_3):
        os.remove(reference_open_eye_3)

    if os.path.isfile(reference_open_eye_4):
        os.remove(reference_open_eye_4)
    
    if os.path.isfile(reference_open_eye_5):
        os.remove(reference_open_eye_5)

    if os.path.isfile(reference_close_eye):
        os.remove(reference_close_eye)

    frame_getter = FrameGetter()
    landmarks_2d_detector = FaceLandmarksDetector()
    blink_detector = BlinkDetector()

    sg.theme('Darkgray10')
    column_image=[
        [
            sg.Image(filename='', key='image',pad=(20,20)),
            sg.VerticalSeparator()
        ]
    ]

    column_sliders =[
        [
            sg.Slider(key="metric",range=(0,2), default_value=1.2,resolution=0.1, size=(12,15), orientation='horizontal', font='Helvetica 11', pad=((10,10),(20,20))),
            sg.Text('Metric', key='_V2_', font='Helvetica 13', pad=(0,0))
        ],
    ]   

    # define the window layout
    layout = [
                [
                    sg.Text('Click Calibration Screen', justification='center', font='Helvetica 25', expand_x=True)
                ],
                [
                    sg.Column(column_image),
                    sg.Column(column_sliders)
                ],
                [
                    [
                        sg.Text('Step:', justification='left', font='Helvetica 14'),
                        sg.Text("Instruction", size=(40, 1),key="instruction", justification='left', font='Helvetica 14'),
                    ],
                    [
                        sg.Text("Eye Status", size=(40, 1),key="eye_open", justification='left', font='Helvetica 14')
                    ]
                ],

                [
                    sg.Button('Save', size=(15, 1), font='Helvetica 14'),
                    sg.Button('Exit', size=(15, 1), font='Helvetica 14'), 
                ],
                [
                    sg.Text('Authors: Turletti, Enzo - Santarelli, Francisco',font='Helvetica 8'),
                    sg.Text('FCEIA - UNR',font='Helvetica 8')
                ]

            ]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(400, 400))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    recording = False

    while True:
        event, values = window.read(timeout=20)

        if state == States.OPEN_EYE_1:
            window['instruction'].update("see screen center and save.")
        if state == States.OPEN_EYE_2:
            window['instruction'].update("see upper right corner of the screen and save.")
        if state == States.OPEN_EYE_3:
            window['instruction'].update("see upper left corner of the screen and save.")
        if state == States.OPEN_EYE_4:
            window['instruction'].update("lower right corner of the screen and save.")
        if state == States.OPEN_EYE_5:
            window['instruction'].update("lower left corner of the screen and save.")
        if state == States.CLOSE_EYE:
            window['instruction'].update("close eye and save.")
        if state == States.METRIC:
            window['instruction'].update("test and tune the metric.")

        metric = int(values["metric"])
        blink_detector.metric = metric
        
        recording = True

        if event == 'Exit' or event == sg.WIN_CLOSED:
            frame_getter.stop()
            return

        if recording:
            frame = frame_getter.get().copy()
            success,landmarks2d = landmarks_2d_detector.detect(frame)

            if success:
                eye = blink_detector.det_right_eye(frame,landmarks2d)

                if event == "Save":
                    if state == States.OPEN_EYE_1:
                        cv2.imwrite(reference_open_eye_1, eye)
                        state = States.OPEN_EYE_2
                        continue
                    if state == States.OPEN_EYE_2:
                        cv2.imwrite(reference_open_eye_2, eye)
                        state = States.OPEN_EYE_3
                        continue
                    if state == States.OPEN_EYE_3:
                        cv2.imwrite(reference_open_eye_3, eye)
                        state = States.OPEN_EYE_4
                        continue
                    if state == States.OPEN_EYE_4:
                        cv2.imwrite(reference_open_eye_4, eye)
                        state = States.OPEN_EYE_5
                        continue
                    if state == States.OPEN_EYE_5:
                        cv2.imwrite(reference_open_eye_5, eye)
                        state = States.CLOSE_EYE
                        continue
                    if state == States.CLOSE_EYE:
                        cv2.imwrite(reference_close_eye, eye)
                        item = {"metric"    :  1.2}

                        with open(output, 'w') as fp:
                            fp.write("%s\n" % item)

                        blink_detector = BlinkDetector()
                        state = States.METRIC
                        continue
                    if state == States.METRIC:
                        item = {"metric"    :  metric}
                        with open(output, 'w') as fp:
                            fp.write("%s\n" % item)
                        
                if state == States.METRIC:
                    open_eye = blink_detector.detect(frame,landmarks2d)
                    if open_eye:
                        window['eye_open'].update(text_color='green')
                    else:
                        window['eye_open'].update(text_color='red')

                imgbytes = cv2.imencode('.png', eye)[1].tobytes()
                window['image'].update(data=imgbytes)

main()