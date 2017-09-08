"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import random

from streams.generators.tools.transition_functions import Transition


class CIRCLES:

    def __init__(self, concept_length=25000, transition_length=500, noise_rate=0.1, random_seed=10):
        self.__CIRCLES = [[[0.2, 0.5], 0.15], [[0.4, 0.5], 0.2], [[0.6, 0.5], 0.25], [[0.8, 0.5], 0.3]]
        self.__INSTANCES_NUM = concept_length * len(self.__CIRCLES)
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_DRIFTS = len(self.__CIRCLES) - 1
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
        return CIRCLES.__name__

    def generate(self, output_path="CIRCLES"):

        random.seed(self.__RANDOM_SEED)

        # [1] CREATING RECORDS
        for i in range(0, self.__INSTANCES_NUM):
            concept_sec = int(i / self.__CONCEPT_LENGTH)
            record = self.create_record(self.__CIRCLES[concept_sec])
            self.__RECORDS.append(list(record))

        # [2] TRANSITION
        for i in range(0, self.__NUM_DRIFTS):
            transition = []
            for j in range(0, self.__W):
                if random.random() < Transition.sigmoid(j, self.__W):
                    record = self.create_record(self.__CIRCLES[i + 1])
                else:
                    record = self.create_record(self.__CIRCLES[i])
                transition.append(list(record))
            starting_index = i * self.__CONCEPT_LENGTH
            ending_index = starting_index + self.__W
            self.__RECORDS[starting_index: ending_index] = transition

        # [3] ADDING NOISE
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        self.write_to_arff(output_path + ".arff")

    def create_record(self, circle):
        x, y, c = self.create_attribute_values(circle)
        if random.random() < 0.5:
            while c == 'p':
                x, y, c = self.create_attribute_values(circle)
        else:
            while c == 'n':
                x, y, c = self.create_attribute_values(circle)
        return x, y, c

    def create_attribute_values(self, circle):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        res = self.get_circle_result(circle[0], x, y, circle[1])
        c = 'p' if res > 0 else 'n'
        return x, y, c

    @staticmethod
    def get_circle_result(c, x, y, radius):
        return (x - c[0])**2 + (y - c[1])**2 - radius**2

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
        arff_writer.write("@relation CIRCLES" + "\n")
        arff_writer.write("@attribute x real" + "\n" +
                          "@attribute y real" + "\n" +
                          "@attribute class {p,n}" + "\n\n")
        arff_writer.write("@data" + "\n")

        for i in range(0, len(self.__RECORDS)):
            arff_writer.write(str("%0.5f" % self.__RECORDS[i][0]) + "," +
                              str("%0.5f" % self.__RECORDS[i][1]) + "," +
                              self.__RECORDS[i][2] + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")
