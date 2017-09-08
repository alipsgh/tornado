"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import os
import time


class Project:
    """This class is used to create a new project."""

    def __init__(self, project_folder, project_name):
        self.__project_path = self.__create(project_folder, project_name)
        self.__project_name = project_name

    @staticmethod
    def __create(project_folder, project_name):
        project_path = project_folder + "/" + project_name
        project_path += "/" + str(time.strftime("%Y%m%d")) + "_" + str(time.strftime("%H%M%S")) + "/"

        if not os.path.exists(project_path):
            os.makedirs(project_path)

        print('The project path "' + project_path + '" is created.')
        return project_path

    def get_path(self):
        return self.__project_path

    def get_name(self):
        return self.__project_name
