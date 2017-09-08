"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import random
from math import *

from streams.generators.tools.transition_functions import Transition


class SINE1:

    def __init__(self, concept_length=20000, transition_length=50, noise_rate=0.1, random_seed=1):
        self.__INSTANCES_NUM = 5 * concept_length
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_DRIFTS = 4
        self.__W = transition_length
        self.__RECORDS = []

        self.__RANDOM_SEED = random_seed
        random.seed(self.__RANDOM_SEED)
        self.__NOISE_LOCATIONS = random.sample(range(0, self.__INSTANCES_NUM), int(self.__INSTANCES_NUM * noise_rate))

        print("You are going to generate a " + self.get_class_name() + " data stream containing " +
              str(self.__INSTANCES_NUM) + " instances, and " + str(self.__NUM_DRIFTS) + " concept drifts; " + "\n\r" +
              "where they appear at every " + str(self.__CONCEPT_LENGTH) + " instances.")

    @staticmethod
    def get_class_name():
        return SINE1.__name__

    def generate(self, output_path="SINE1"):

        random.seed(self.__RANDOM_SEED)

        # [1] CREATING RECORDS
        for i in range(0, self.__INSTANCES_NUM):
            concept_sec = int(i / self.__CONCEPT_LENGTH)
            dist_id = int(concept_sec % 2)
            record = self.create_record(dist_id)
            self.__RECORDS.append(list(record))

        # [2] TRANSITION
        for i in range(1, self.__NUM_DRIFTS + 1):
            transition = []
            if (i % 2) == 1:
                for j in range(0, self.__W):
                    if random.random() < Transition.sigmoid(j, self.__W):
                        record = self.create_record(1)
                    else:
                        record = self.create_record(0)
                    transition.append(list(record))
            else:
                for j in range(0, self.__W):
                    if random.random() < Transition.sigmoid(j, self.__W):
                        record = self.create_record(0)
                    else:
                        record = self.create_record(1)
                    transition.append(list(record))
            starting_index = i * self.__CONCEPT_LENGTH
            ending_index = starting_index + self.__W
            self.__RECORDS[starting_index: ending_index] = transition

        # [3] ADDING NOISE
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        self.write_to_arff(output_path + ".arff")

    def create_record(self, dist_id):
        x, y, c = self.create_attribute_values()
        if random.random() < 0.5:
            while c != 'p':
                x, y, c = self.create_attribute_values()
        else:
            while c != 'n':
                x, y, c = self.create_attribute_values()
        if dist_id == 1:
            c = 'n' if c == 'p' else 'p'
        return x, y, c

    @staticmethod
    def create_attribute_values():
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        c = 'p' if y <= sin(x) else 'n'
        return x, y, c

    def add_noise(self):
        for i in range(0, len(self.__NOISE_LOCATIONS)):
            noise_spot = self.__NOISE_LOCATIONS[i]
            c = self.__RECORDS[noise_spot][2]
            if c == 'p':
                self.__RECORDS[noise_spot][2] = 'n'
            else:
                self.__RECORDS[noise_spot][2] = 'p'

    def write_to_arff(self, output_path):
        arff_writer = open(output_path, "w")
        arff_writer.write("@relation SINE1" + "\n")
        arff_writer.write("@attribute x real" + "\n" +
                          "@attribute y real" + "\n" +
                          "@attribute class {p,n}" + "\n\n")
        arff_writer.write("@data" + "\n")
        for i in range(0, len(self.__RECORDS)):
            arff_writer.write(str("%0.3f" % self.__RECORDS[i][0]) + "," +
                              str("%0.3f" % self.__RECORDS[i][1]) + "," +
                              self.__RECORDS[i][2] + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")
