from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tkinter as tk
from PIL import Image, ImageTk


class ImageWindow(object):
    def __init__(self, image_path, title):
        self.image_path = image_path
        self.root = None
        self.title = title
        self.panel = None
        self.image = None

    def show_image(self):
        self.root = tk.Tk()
        self.root.title(self.title)

        # pick an image file you have .bmp  .jpg  .gif.  .png
        # load the file and covert it to a Tkinter image object
        self.image = ImageTk.PhotoImage(Image.open(self.image_path))

        # get the image size
        w = self.image.width()
        h = self.image.height()

        # make the root window the size of the image
        self.root.geometry("%dx%d" % (w, h))

        # root has no image argument, so use a label as a panel
        self.panel = tk.Label(self.root, image=self.image)
        self.panel.pack(side="bottom", fill="both", expand="yes")
        self.root.update()

    def update_image(self):
        self.image = ImageTk.PhotoImage(Image.open(self.image_path))
        self.panel.configure(image=self.image)
        # get the image size
        w = self.image.width()
        h = self.image.height()

        # make the root window the size of the image
        self.root.geometry("%dx%d" % (w, h))
        self.root.update()


def show_image(image_path, title):
    imagew = ImageWindow(image_path, title)
    imagew.show_image()
    return imagew
