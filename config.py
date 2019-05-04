"""Configuration variables"""

import os
cwd = os.path.dirname(os.path.realpath(__file__))

data_directory = cwd + os.path.sep + os.path.normpath("data") + os.path.sep
download_directory = cwd + os.path.sep + os.path.normpath("downloads") + os.path.sep
images_directory = cwd + os.path.sep + os.path.normpath("Images") + os.path.sep
