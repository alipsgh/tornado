"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import copy
import math
import random

from streams.generators.tools.transition_functions import Transition


class LED:

    def __init__(self, num_instances, num_irr=0, num_attr_drifts=0):

        self.SEGMENTS_LED = [
            [[1, 1, 1, 1, 1, 1, 0], 0],
            [[0, 0, 0, 0, 1, 1, 0], 1],
            [[1, 0, 1, 1, 0, 1, 1], 2],
            [[1, 0, 0, 1, 1, 1, 1], 3],
            [[0, 1, 0, 0, 1, 1, 1], 4],
            [[1, 1, 0, 1, 1, 0, 1], 5],
            [[1, 1, 1, 1, 1, 0, 1], 6],
            [[1, 0, 0, 0, 1, 1, 0], 7],
            [[1, 1, 1, 1, 1, 1, 1], 8],
            [[1, 1, 0, 0, 1, 1, 1], 9]
        ]

        self.NUM_INSTANCES = num_instances
        self.NUM_IRRELEVANT_ATTRIBUTE = num_irr
        self.add_irrelevant_attributes()
        self.drift_attributes(num_attr_drifts)

    def add_irrelevant_attributes(self):
        for i in range(0, self.NUM_IRRELEVANT_ATTRIBUTE):
            for j in range(0, 10):
                self.SEGMENTS_LED[j][0].append(None)

    def drift_attributes(self, num_attr_drifts):
        for i in range(0, num_attr_drifts):
            attr_1 = random.randint(0, 6)
            attr_2 = random.randint(0, 6 + self.NUM_IRRELEVANT_ATTRIBUTE)
            while attr_2 == attr_1:
                attr_2 = random.randint(0, 6 + self.NUM_IRRELEVANT_ATTRIBUTE)
            for j in range(0, 10):
                tmp_var = self.SEGMENTS_LED[j][0][attr_1]
                self.SEGMENTS_LED[j][0][attr_1] = self.SEGMENTS_LED[j][0][attr_2]
                self.SEGMENTS_LED[j][0][attr_2] = tmp_var

    def generate(self):
        records = []
        for i in range(0, self.NUM_INSTANCES):
            records.append(self.create_instance())
        return records

    def create_instance(self):
        digit = int(math.floor(random.uniform(0, 10)))
        instance = copy.copy(self.SEGMENTS_LED[digit])
        instance_x = copy.copy(instance[0])
        instance_y = instance[1]
        for i in range(0, len(instance_x)):
            if instance_x[i] is None:
                r = random.randint(0, 1)
                instance_x[i] = r
        return [instance_x, instance_y]


class LEDConceptDrift:

    def __init__(self, concept_length=25000, num_irr_attr=17, led_attr_drift=[0, 3, 1, 3],
                 transition_length=500, noise_rate=0.1, random_seed=1):

        self.__INSTANCES_NUM = concept_length * len(led_attr_drift)
        self.__CONCEPT_LENGTH = concept_length
        self.__NUM_IRR_ATTR = num_irr_attr
        self.__LED_ATTR_DRIFTS = led_attr_drift
        self.__NUM_DRIFTS = len(led_attr_drift) - 1
        self.__W = transition_length
        self.__RECORDS = []

        self.__RANDOM_SEED = random_seed
        random.seed(self.__RANDOM_SEED)
        self.__NOISE_LOCATIONS = random.sample(range(0, self.__INSTANCES_NUM), int(self.__INSTANCES_NUM * noise_rate))

        self.LED_GENERATORS = self.create_led_objects()

        print("You are going to generate a " + self.get_class_name() + " data stream containing " +
              str(self.__INSTANCES_NUM) + " instances, and " + str(self.__NUM_DRIFTS) + " concept drifts; " + "\n\r" +
              "where they appear at every " + str(self.__CONCEPT_LENGTH) + " instances.")

    @staticmethod
    def get_class_name():
        return LEDConceptDrift.__name__

    def create_led_objects(self):
        led_objects = []
        for i in range(0, self.__NUM_DRIFTS + 1):
            led_objects.append(LED(self.__CONCEPT_LENGTH, self.__NUM_IRR_ATTR, self.__LED_ATTR_DRIFTS[i]))
        return led_objects

    def generate(self, output_path="LED"):

        random.seed(self.__RANDOM_SEED)

        # [1] CREATING RECORDS
        for led in self.LED_GENERATORS:
            self.__RECORDS += led.generate()

        # [2] TRANSITION
        if self.__W != 0:
            for i in range(0, len(self.LED_GENERATORS) - 1):
                transition = []
                for j in range(0, self.__W):
                    if random.random() < Transition.sigmoid(j, self.__W):
                        transition.append(self.LED_GENERATORS[i + 1].create_instance())
                    else:
                        transition.append(self.LED_GENERATORS[i].create_instance())
                starting_index = (i + 1) * self.__CONCEPT_LENGTH
                ending_index = (i + 1) * self.__CONCEPT_LENGTH + self.__W
                self.__RECORDS[starting_index:ending_index] = transition

        # [3] ADDING NOISE
        if len(self.__NOISE_LOCATIONS) != 0:
            self.add_noise()

        self.write_to_arff(output_path + ".arff")

    def add_noise(self):
        for i in range(0, len(self.__NOISE_LOCATIONS)):
            noise_spot = self.__NOISE_LOCATIONS[i]
            y_r = self.__RECORDS[noise_spot][1]
            y_n = random.randint(0, 9)
            while y_n == y_r:
                y_n = random.randint(0, 9)
            self.__RECORDS[noise_spot][1] = y_n

    def write_to_arff(self, output_path):
        arff_writer = open(output_path, "w")
        arff_writer.write("@relation LED" + "\n\n")
        attributes_str = ""
        for a in range(0, 7 + self.__NUM_IRR_ATTR):
            attributes_str += "@attribute a" + str(a) + " {0,1}" + "\n"
        arff_writer.write(attributes_str + "\n" + "@attribute class {0,1,2,3,4,5,6,7,8,9}" + "\n\n")
        arff_writer.write("@data" + "\n")
        for r in self.__RECORDS:
            x = r[0]
            y = r[1]
            x_str = ''
            for x_i in x:
                    x_str = x_str + str(x_i) + ","
            arff_writer.write(x_str + str(y) + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")
