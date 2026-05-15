import numpy as np
import pytest

from drift_detection.mddm_a import MDDM_A
from drift_detection.mddm_e import MDDM_E
from drift_detection.mddm_g import MDDM_G

np.random.seed(12345)
data_stream_1 = np.concatenate(
    (np.random.randint(2, size=1000), np.random.randint(8, size=1000))
)

np.random.seed(12345)
data_stream_2 = np.concatenate(
    [
        [np.random.binomial(1, 0.2) for _ in range(1000)],
        [np.random.binomial(1, 0.8) for _ in range(1000)],
    ]
).astype(int)

np.random.seed(12345)
data_stream_3 = np.concatenate(
    (
        np.random.normal(0.0, 0.1, 500) > 0,
        np.random.normal(0.25, 0.1, 500) > 0,
        np.random.normal(0.0, 0.1, 500) > 0,
        np.random.normal(0.25, 0.1, 500) > 0,
    )
).astype(int)


def test_mddm_a():
    mddm_a = MDDM_A()
    expected_indices = [1032]
    detected_indices = perform_test(mddm_a, data_stream_2)
    print(detected_indices)
    print(
        "Assertion for MDDM_A with data_stream_2:", detected_indices == expected_indices
    )
    #  assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    mddm_a = MDDM_A()
    expected_indices = [519, 1513]
    detected_indices = perform_test(mddm_a, data_stream_3)
    print(detected_indices)
    print(
        "Assertion for MDDM_A with data_stream_3:", detected_indices == expected_indices
    )
    #  assert detected_indices == expected_indices


def test_mddm_e():
    mddm_e = MDDM_E()
    expected_indices = [1030]
    detected_indices = perform_test(mddm_e, data_stream_2)
    print(detected_indices)
    print(
        "Assertion for MDDM_E with data_stream_2:", detected_indices == expected_indices
    )
    #  assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    mddm_e = MDDM_E()
    expected_indices = [516, 1512]
    detected_indices = perform_test(mddm_e, data_stream_3)
    print(detected_indices)
    print(
        "Assertion for MDDM_E with data_stream_3:", detected_indices == expected_indices
    )
    #  assert detected_indices == expected_indices


def test_mddm_g():
    mddm_g = MDDM_G()
    expected_indices = [1030]
    detected_indices = perform_test(mddm_g, data_stream_2)
    print(detected_indices)
    print(
        "Assertion for MDDM_G with data_stream_2:", detected_indices == expected_indices
    )
    #  assert detected_indices == expected_indices

    # Second test, more abrupt drifts
    mddm_g = MDDM_G()
    expected_indices = [516, 1512]
    detected_indices = perform_test(mddm_g, data_stream_3)
    print(detected_indices)
    print(
        "Assertion for MDDM_G with data_stream_3:", detected_indices == expected_indices
    )
    #  assert detected_indices == expected_indices


def perform_test(drift_detector, data_stream):
    detected_indices = []
    for i, val in enumerate(data_stream):
        # if val == 0:
        #     val = 1
        # elif val == 1:
        #     val = 0
        # else:
        #     raise ValueError(f"Unexpected value in data stream: {val}")
        _, drift_status = drift_detector.detect(val)
        if drift_status:
            detected_indices.append(i)
    return detected_indices


if __name__ == "__main__":
    test_mddm_a()
    test_mddm_e()
    test_mddm_g()
    print("All tests passed.")
