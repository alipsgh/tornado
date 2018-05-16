"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import re

from data_structures.attribute import Attribute
from dictionary.tornado_dictionary import TornadoDic


class ARFFReader:
    """This class is used to read a .arff file."""

    @staticmethod
    def read(file_path):
        labels = []
        attributes = []
        attributes_min_max = []
        records = []
        data_flag = False
        reader = open(file_path, "r")
        for line in reader:
            
            if line.strip() == '':
                continue
            
            if line.startswith("@attribute") or line.startswith("@ATTRIBUTE"):

                line = line.strip('\n\r\t')
                line = line.split(' ')

                attribute_name = line[1]
                attribute_value_range = line[2]

                attribute = Attribute()
                attribute.set_name(attribute_name)
                if attribute_value_range.lower() in ['numeric', 'real', 'integer']:
                    attribute_type = TornadoDic.NUMERIC_ATTRIBUTE
                    attribute_value_range = []
                    attributes_min_max.append([0, 0])
                else:
                    attribute_type = TornadoDic.NOMINAL_ATTRIBUTE
                    attribute_value_range = attribute_value_range.strip('{}').replace("'", "")
                    attribute_value_range = attribute_value_range.split(',')
                    attributes_min_max.append([None, None])
                attribute.set_type(attribute_type)
                attribute.set_possible_values(attribute_value_range)

                attributes.append(attribute)

            elif line.startswith("@data") or line.startswith("@DATA"):
                data_flag = True
                labels = attributes[len(attributes) - 1].POSSIBLE_VALUES
                attributes.pop(len(attributes) - 1)
                continue

            elif data_flag is True:
                line = re.sub('\s+', '', line)
                elements = line.split(',')
                for i in range(0, len(elements) - 1):
                    if attributes[i].TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                        elements[i] = float(elements[i])
                        min_value = attributes_min_max[i][0]
                        max_value = attributes_min_max[i][1]
                        if elements[i] < min_value:
                            min_value = elements[i]
                        elif elements[i] > max_value:
                            max_value = elements[i]
                        attributes_min_max[i] = [min_value, max_value]
                records.append(elements)

        for i in range(0, len(attributes)):
            if attributes[i].TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                attributes[i].set_bounds_values(attributes_min_max[i][0], attributes_min_max[i][1])

        return labels, attributes, records
