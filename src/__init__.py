# __init__.py

import os


class PathInit:
  def __init__(self, name):
    self.name = name

  def get_root(self):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return project_root
