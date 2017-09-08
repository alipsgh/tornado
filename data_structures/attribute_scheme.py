import copy

from dictionary.tornado_dictionary import TornadoDic
from filters.attribute_handlers import Discretizer, NominalToNumericTransformer


class AttributeScheme:

    @staticmethod
    def get_scheme(attributes):

        numeric_attribute_scheme = []
        nominal_attribute_scheme = []

        for a in attributes:
            if a.TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                numeric_attribute_scheme.append(copy.copy(a))
                # NOW LET'S MAKE A COPY FROM THE ATTRIBUTE OBJECT AND DISCRETIZE IT
                discretized_a = copy.copy(a)
                Discretizer.bin_attribute(discretized_a, 10)
                nominal_attribute_scheme.append(discretized_a)
            else:
                nominal_attribute_scheme.append(copy.copy(a))
                # NOW LET'S MAKE A COPY FROM THE ATTRIBUTE OBJECT AND MAKE IT NUMERIC
                numeric_a = copy.copy(a)
                NominalToNumericTransformer.convert_attribute_scheme(numeric_a)
                numeric_attribute_scheme.append(numeric_a)

        return {'numeric': numeric_attribute_scheme, 'nominal': nominal_attribute_scheme}

