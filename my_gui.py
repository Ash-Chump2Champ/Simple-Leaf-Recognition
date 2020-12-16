from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast

from kivy.uix.widget import Widget
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.theming import ThemeManager

from kivymd.uix.label import MDLabel
from kivy.properties import StringProperty
from PIL import Image

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import mahotas as mt

import backend

directory = ''

Builder.load_string("""
<ScreenManagement>:
    ScreenOne:
        name: "menu"
    ScreenTwo:
        name: "filemanager"
    ScreenThree:
        name: "imagedisplay"

<ScreenOne>:

    BoxLayout:
        orientation: 'vertical'
        pos: self.pos
        size: self.size

        MDToolbar:
            title: "Plant Leaf Identificaton using KNN"
            left_action_items: [['leaf', lambda x: None]]
            elevation: 10

        FloatLayout:

            MDRoundFlatIconButton:
                text: "Camera"
                icon: "camera"
                height: 48
                width: 128
                pos_hint: {'center_x': 0.5, 'center_y': 0.6}
                

            MDRoundFlatIconButton:
                text: "Open File"
                icon: "image"
                height: 48
                width: 128
                pos_hint: {'center_x': 0.5, 'center_y': 0.45}
                on_press: root.manager.current= 'filemanager'

<ScreenTwo>:
    BoxLayout:
        orientation: 'vertical'
        pos: self.pos
        size: self.size

        MDToolbar:
            title: "Plant Leaf Identificaton using KNN"
            icon: "leaf"
            left_action_items: [['leaf', lambda x: None]]
            elevation: 10

        FloatLayout:

            MDRectangleFlatIconButton:
                text: "Open manager"
                icon: "folder"
                pos_hint: {'center_x': 0.5, 'center_y': 0.8}
                on_release: root.file_manager_open()

            MDLabel:
                text: str(root.photo)
                pos_hint: {'center_x': 0.5, 'center_y': 0.7}
                halign: "center"

            MDRectangleFlatIconButton:
                text: "Next"
                icon: "magnify-scan"
                width: 12
                pos_hint: {'center_x': 0.5, 'center_y': 0.3}
                on_release: root.manager.current = 'imagedisplay'

            MDRoundFlatIconButton:
                text: "Go Back"
                icon: "page-previous"
                pos_hint: {'center_x': 0.5, 'center_y': 0.15}
                on_release: root.manager.current = 'menu'

<ScreenThree>:
    BoxLayout:
        orientation: 'vertical'
        pos: self.pos
        size: self.size

        MDToolbar:
            title: "Plant Leaf Identificaton using KNN"
            icon: "leaf"
            left_action_items: [['leaf', lambda x: None]]
            elevation: 10

        FloatLayout:

            Image:
                id: image
                size_hint: None,None
                size: 200, 200
                pos_hint: {'center_x': 0.5, 'center_y': 0.75}
            
            MDLabel:
                id:plantname
                text: str(root.my_plant)
                pos_hint: {'center_x': 0.5, 'center_y': 0.5}
                halign: "center"

""")

class ScreenOne(Screen):
    theme_cls = ThemeManager()
    theme_cls.primary_palette = 'Green'
    main_widget = None

class ScreenTwo(Screen):
    photo = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            previous=True,
        )

    def file_manager_open(self):
        self.file_manager.show('/users/lenovo/pictures')  # output manager to the screen
        self.manager_open = True

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''
        global directory
        self.exit_manager()
        directory = 'C:' + path
        self.photo = directory

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True


class ScreenThree(Screen):
    my_plant = StringProperty('')

    def on_pre_enter(self, *args):
        global directory
        imageWidget = self.ids['image']
        imageWidget.source = directory
        analyser = backend.driverprogram()
        analyser.train_file(directory)

    def plant_name(self,plant):
        var1 = str(plant)
        var2 = var1.replace('[\'','')
        var3 = var2.replace('\']','')
        var4 = 'Plant Name: '+ var3
        print(var4)
        self.my_plant = var4

class ScreenManagement(ScreenManager):
    pass

class Interface(MDApp):

    def build(self):
        self.theme_cls = ThemeManager()
        self.theme_cls.primary_palette = 'Green'
        return ScreenManagement()
