import os.path
import src.nn.config as config

path = os.getcwd()
config.ROOT_PATH = path[:path.index('/src')]
