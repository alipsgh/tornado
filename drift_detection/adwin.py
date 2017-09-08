"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The ADaptive WINdowing (ADWIN) Method Implementation ***
Paper: Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."
Published in: Proceedings of the 2007 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2007.
URL: http://www.cs.upc.edu/~GAVALDA/papers/adwin06.pdf
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class ListItem:

    def __init__(self, next_node=None, previous_node=None):

        self.bucket_size_row = 0
        self.MAXBUCKETS = 5

        self.bucket_total = []
        self.bucket_variance = []
        for i in range(0, self.MAXBUCKETS + 1):
            self.bucket_total.append(0)
            self.bucket_variance.append(0)

        self.next = next_node
        self.previous = previous_node
        if next_node is not None:
            next_node.previous = self
        if previous_node is not None:
            previous_node.next = self

        self.clear()

    def clear(self):
        self.bucket_size_row = 0
        for k in range(0, self.MAXBUCKETS + 1):
            self.clear_bucket(k)

    def clear_bucket(self, k):
        self.set_total(0, k)
        self.set_variance(0, k)

    def insert_bucket(self, value, variance):
        k = self.bucket_size_row
        self.bucket_size_row += 1
        self.set_total(value, k)
        self.set_variance(variance, k)

    def remove_bucket(self):
        self.compress_buckets_row(1)

    def compress_buckets_row(self, number_items_deleted):
        for k in range(number_items_deleted, self.MAXBUCKETS + 1):
            self.bucket_total[k - number_items_deleted] = self.bucket_total[k]
            self.bucket_variance[k - number_items_deleted] = self.bucket_variance[k]
        for k in range(1, number_items_deleted + 1):
            self.clear_bucket(self.MAXBUCKETS - k + 1)
        self.bucket_size_row -= number_items_deleted

    def set_previous(self, previous_node):
        self.previous = previous_node

    def set_next(self, next_node):
        self.next = next_node

    def get_total(self, k):
        return self.bucket_total[k]

    def set_total(self, value, k):
        self.bucket_total[k] = value

    def get_variance(self, k):
        return self.bucket_variance[k]

    def set_variance(self, value, k):
        self.bucket_variance[k] = value


class List:

    def __init__(self):
        self.count = None
        self.head = None
        self.tail = None

        self.clear()
        self.add_to_head()

    def is_empty(self):
        return self.count == 0

    def clear(self):
        self.head = None
        self.tail = None
        self.count = 0

    def add_to_head(self):
        self.head = ListItem(self.head, None)
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def remove_from_head(self):
        self.head = self.head.next
        if self.head is not None:
            self.head.set_previous(None)
        else:
            self.tail = None
        self.count -= 1

    def add_to_tail(self):
        self.tail = ListItem(None, self.tail)
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self):
        self.tail = self.tail.previous
        if self.tail is None:
            self.head = None
        else:
            self.tail.set_next(None)
        self.count -= 1


class ADWINChangeDetector(SuperDetector):
    """The ADaptive WINdowing (ADWIN) drift detection method class."""

    DETECTOR_NAME = TornadoDic.ADWIN

    def __init__(self, delta=0.002):

        super().__init__()

        self.DELTA = delta
        self.adwin = ADWIN(self.DELTA)

    def run(self, pr):
        drift_status = self.adwin.set_input(pr)
        return False, drift_status

    def reset(self):
        super().reset()
        self.adwin = ADWIN(self.DELTA)

    def get_settings(self):
        return [str(self.DELTA),
                "$\delta$:" + str(self.DELTA).upper()]


class ADWIN:

    def __init__(self, delta):

        self.DELTA = delta

        self.mint_minim_longitud_window = 10
        self.mint_time = 0
        self.mint_clock = 32

        self.last_bucket_row = 0

        self.bucket_number = 0
        self.detect = 0
        self.detect_twice = 0
        self.mint_min_win_length = 5

        self.MAXBUCKETS = 5
        self.TOTAL = 0
        self.VARIANCE = 0
        self.WIDTH = 0

        self.list_row_buckets = List()

    def insert_element(self, value):
        self.WIDTH += 1
        self.insert_element_bucket(0, value, self.list_row_buckets.head)
        inc_variance = 0
        if self.WIDTH > 1:
            inc_variance = (self.WIDTH - 1) * (value - self.TOTAL / (self.WIDTH - 1)) * (value - self.TOTAL / (self.WIDTH - 1)) / self.WIDTH
        self.VARIANCE += inc_variance

        self.TOTAL += value
        self.compress_buckets()

    def insert_element_bucket(self, variance, value, node):
        node.insert_bucket(value, variance)
        self.bucket_number += 1

    @staticmethod
    def bucket_size(row):
        return int(pow(2, row))

    def delete_element(self):
        node = self.list_row_buckets.tail
        n1 = self.bucket_size(self.last_bucket_row)
        self.WIDTH -= n1
        self.TOTAL -= node.get_total(0)
        u1 = node.get_total(0) / n1
        inc_variance = node.get_variance(0) + n1 * self.WIDTH * (u1 - self.TOTAL / self.WIDTH) * (u1 - self.TOTAL / self.WIDTH) / (n1 + self.WIDTH)
        self.VARIANCE -= inc_variance
        if self.VARIANCE < 0:
            self.VARIANCE = 0

        node.remove_bucket()
        self.bucket_number -= 1
        if node.bucket_size_row == 0:
            self.list_row_buckets.remove_from_tail()
            self.last_bucket_row -= 1
        return n1

    def compress_buckets(self):
        cursor = self.list_row_buckets.head
        i = 0
        while True:
            k = cursor.bucket_size_row
            if k == self.MAXBUCKETS + 1:
                next_node = cursor.next
                if next_node is None:
                    self.list_row_buckets.add_to_tail()
                    next_node = cursor.next
                    self.last_bucket_row += 1
                n1 = self.bucket_size(i)
                n2 = self.bucket_size(i)
                u1 = cursor.get_total(0) / n1
                u2 = cursor.get_total(1) / n2
                inc_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                next_node.insert_bucket(cursor.get_total(0) + cursor.get_total(1), cursor.get_variance(0) + cursor.get_variance(1) + inc_variance)
                self.bucket_number += 1
                cursor.compress_buckets_row(2)
                if next_node.bucket_size_row <= self.MAXBUCKETS:
                    break
            else:
                break
            cursor = cursor.next
            i += 1
            if cursor is None:
                break

    def set_input(self, pr):
        bln_change = False
        self.mint_time += 1
        self.insert_element(pr)

        if self.mint_time % self.mint_clock == 0 and self.WIDTH > self.mint_minim_longitud_window:
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = False
                bln_exit = False
                n0 = 0
                n1 = self.WIDTH
                u0 = 0
                u1 = self.TOTAL

                cursor = self.list_row_buckets.tail
                i = self.last_bucket_row
                while True:

                    for k in range(0, cursor.bucket_size_row):

                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)

                        if i == 0 and k == cursor.bucket_size_row - 1:
                            bln_exit = True
                            break

                        if n1 > self.mint_min_win_length + 1 and n0 > self.mint_min_win_length + 1 and self.bln_cut_expression(n0, n1, u0, u1):
                            self.detect = self.mint_time

                            if self.detect == 0:
                                self.detect = self.mint_time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.mint_time
                            bln_reduce_width = True
                            bln_change = True
                            if self.WIDTH > 0:
                                n0 -= self.delete_element()
                                bln_exit = True
                                break
                    cursor = cursor.previous
                    i -= 1
                    if not (not bln_exit and cursor is not None):
                        break

        return bln_change

    def bln_cut_expression(self, n0, n1, u0, u1):
        diff = math.fabs((u0 / n0) - (u1 / n1))
        n = self.WIDTH
        m = (1 / (n0 - self.mint_min_win_length + 1)) + (1 / (n1 - self.mint_min_win_length + 1))
        dd = math.log(2 * math.log(n) / self.DELTA)
        v = self.VARIANCE / self.WIDTH
        e = math.sqrt(2 * m * v * dd) + 2 / 3 * dd * m
        return diff > e
