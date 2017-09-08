"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import random

from streams.generators.tools.transition_functions import Transition


class STAGGER:

    def __init__(self, concept_length=33334, transition_length=50, noise_rate=0.1, random_seed=1):
        self.__NUM_INSTANCES = 3 * concept_length - 2
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_DRIFTS = 2
        self.__SIZE = ['small', 'medium', 'large']
        self.__COLOR = ['red', 'green']
        self.__SHAPE = ['circular', 'non-circular']
        self.__W = transition_length
        self.__RECORDS = []

        self.__RANDOM_SEED = random_seed
        random.seed(self.__RANDOM_SEED)
        self.__NOISE_LOCATIONS = random.sample(range(0, self.__NUM_INSTANCES), int(self.__NUM_INSTANCES * noise_rate))

        print("You are going to generate a " + self.get_class_name() + " data stream containing " +
              str(self.__NUM_INSTANCES) + " instances, and " + str(self.__NUM_DRIFTS) + " concept drifts; " + "\n\r" +
              "where they appear at every " + str(self.__CONCEPT_LENGTH) + " instances.")

    @staticmethod
    def get_class_name():
        return STAGGER.__name__

    def generate(self, output_path="STAGGER"):

        random.seed(self.__RANDOM_SEED)

        # [1] CREATING RECORDS
        for i in range(0, self.__NUM_INSTANCES):
            context_id = int(i / self.__CONCEPT_LENGTH)
            record = self.create_record(context_id)
            self.__RECORDS.append(list(record))

        # [2] TRANSITION
        for i in range(1, self.__NUM_DRIFTS + 1):
            transition = []
            for j in range(0, self.__W):
                if random.random() < Transition.sigmoid(j, self.__W):
                    record = self.create_record(i)
                else:
                    record = self.create_record(i - 1)
                transition.append(list(record))
            starting_index = i * self.__CONCEPT_LENGTH
            ending_index = starting_index + self.__W
            self.__RECORDS[starting_index:ending_index] = transition

        # [3] ADDING NOISE
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        self.write_to_arff(output_path + ".arff")

    def create_record(self, context_id):
        size, color, shape, c = self.create_attribute_values(context_id)
        if random.random() < 0.5:
            while c == 'p':
                size, color, shape, c = self.create_attribute_values(context_id)
        else:
            while c == 'n':
                size, color, shape, c = self.create_attribute_values(context_id)
        return size, color, shape, c

    def create_attribute_values(self, context_id):
        size = self.__SIZE[random.randint(0, 2)]
        color = self.__COLOR[random.randint(0, 1)]
        shape = self.__SHAPE[random.randint(0, 1)]
        res = self.get_stagger_result(size, color, shape, context_id)
        c = 'p' if res is True else 'n'
        return size, color, shape, c

    def get_stagger_result(self, size, color, shape, context_id):
        if context_id == 0:
            if size == self.__SIZE[0] and color == self.__COLOR[0]:
                return True
        elif context_id == 1:
            if color == self.__COLOR[1] or shape == self.__SHAPE[0]:
                return True
        elif context_id == 2:
            if size == self.__SIZE[1] or size == self.__SIZE[2]:
                return True
        else:
            return False

    def add_noise(self):
        for i in range(0, len(self.__NOISE_LOCATIONS)):
            noise_spot = self.__NOISE_LOCATIONS[i]
            c = self.__RECORDS[3]
            if c == 'p':
                self.__RECORDS[noise_spot][3] = 'n'
            else:
                self.__RECORDS[noise_spot][3] = 'p'

    def write_to_arff(self, output_path):
        arff_writer = open(output_path, "w")
        arff_writer.write("@relation 'STAGGER'" + "\n")
        arff_writer.write("@attribute 'size' {small,medium,large}" + "\n" +
                          "@attribute 'color' {red,green}" + "\n" +
                          "@attribute 'shape' {circular,non-circular}" + "\n" +
                          "@attribute 'class' {p,n}" + "\n\n")
        arff_writer.write("@data" + "\n")

        for record in self.__RECORDS:
            for i in range(0, len(record)):
                if i != len(record) - 1:
                    arff_writer.write(record[i] + ",")
                else:
                    arff_writer.write(record[i] + "\n")

        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")
