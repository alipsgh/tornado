"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import os
import zipfile
from os.path import basename


class Archiver:
    """
    This class stores results of experiments in .zip files for future reference!
    """

    @staticmethod
    def archive_single(label, stats, dir_path, name, sub_name):

        file_path = (dir_path + name + "_" + sub_name).lower()

        stats_writer = open(file_path + ".txt", 'w')
        stats_writer.write(label + "\n")
        stats_writer.write(str(stats) + "\n")
        stats_writer.close()

        zipper = zipfile.ZipFile(file_path + ".zip", 'w')
        zipper.write(file_path + ".txt", compress_type=zipfile.ZIP_DEFLATED, arcname=basename(file_path + ".txt"))
        zipper.close()

        os.remove(file_path + ".txt")

    @staticmethod
    def archive_multiple(labels, stats, dir_path, name, sub_name):

        file_path = (dir_path + name + "_" + sub_name).lower()

        stats_writer = open(file_path + ".txt", 'w')
        for i in range(0, len(labels)):
            stats_writer.write(labels[i] + "\n")
            stats_writer.write(str(stats[i]) + "\n")
        stats_writer.close()

        zipper = zipfile.ZipFile(file_path + ".zip", 'w')
        zipper.write(file_path + ".txt", compress_type=zipfile.ZIP_DEFLATED, arcname=basename(file_path + ".txt"))
        zipper.close()

        os.remove(file_path + ".txt")
