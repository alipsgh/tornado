"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import operator
from collections import OrderedDict

from dictionary.tornado_dictionary import TornadoDic


class NominalToNumericTransformer:
    """This is a nominal to numeric scheme transformer."""

    @staticmethod
    def convert_attribute_scheme(attribute):
        possible_values = OrderedDict()
        for i in range(1, len(attribute.POSSIBLE_VALUES) + 1):
            # if attribute.POSSIBLE_VALUES[i - 1].isdigit() and len(attribute.POSSIBLE_VALUES) == 2:
            #    possible_values[attribute.POSSIBLE_VALUES[i - 1]] = float(attribute.POSSIBLE_VALUES[i - 1])
            #else:
            possible_values[attribute.POSSIBLE_VALUES[i - 1]] = i
        attribute.TYPE = TornadoDic.NUMERIC_ATTRIBUTE
        attribute.POSSIBLE_VALUES = possible_values
        attribute.MAXIMUM_VALUE = max(possible_values.items(), key=operator.itemgetter(1))[1]
        attribute.MINIMUM_VALUE = min(possible_values.items(), key=operator.itemgetter(1))[1]

    @staticmethod
    def map_attribute_value(x_index, attribute_scheme):
        return attribute_scheme.POSSIBLE_VALUES[x_index]


class NumericToNominalTransformer:
    """This is a numeric to nominal scheme transformer."""

    @staticmethod
    def convert_attribute_scheme(attribute):
        possible_values = []
        for v in attribute.POSSIBLE_VALUES:
            possible_values.append(str(v))
        attribute.TYPE = TornadoDic.NOMINAL_ATTRIBUTE
        attribute.POSSIBLE_VALUES = possible_values

    @staticmethod
    def map_attribute_value(x):
        return str(x)


class Normalizer:
    """The min-max normalizer."""

    @staticmethod
    def normalize(record, scheme):
        for i in range(0, len(record)):
            record[i] = (record[i] - scheme[i].MINIMUM_VALUE) / (scheme[i].MAXIMUM_VALUE - scheme[i].MINIMUM_VALUE)
        return record


class Discretizer:
    """The bin-based discretizer."""

    @staticmethod
    def bin_attribute(attribute, num_of_bins):
        bins = []
        w = (attribute.MAXIMUM_VALUE - attribute.MINIMUM_VALUE) / num_of_bins
        for k in range(0, num_of_bins):
            lower_bound = round(attribute.MINIMUM_VALUE + (k * w), 10)
            upper_bound = round(attribute.MINIMUM_VALUE + ((k + 1) * w), 10)
            bins.append(str(lower_bound) + '..' + str(upper_bound))
        attribute.TYPE = TornadoDic.NOMINAL_ATTRIBUTE
        attribute.set_possible_values(bins)

    @staticmethod
    def find_bin(x, attribute):
        for v in attribute.POSSIBLE_VALUES:
            upper_bound = float(v.split("..")[1])
            if x <= upper_bound:
                x = v
                break
        return x
