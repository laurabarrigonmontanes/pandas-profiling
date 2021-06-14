import numpy as np
import pandas as pd
import pytest

from pandas_profiling.model.dataframe_wrappers import SparkSeries
from pandas_profiling.model.summary_algorithms import (
    describe_boolean_1d,
    describe_boolean_spark_1d,
    describe_counts,
    describe_timestamp_spark_1d,
)


def test_count_summary_sorted():
    s = pd.Series([1] + [2] * 1000)
    sn, r = describe_counts(s, {})
    assert r["value_counts_without_nan"].index[0] == 2
    assert r["value_counts_without_nan"].index[1] == 1


def test_count_summary_nat():
    s = pd.to_datetime(pd.Series([1, 2] + [np.nan, pd.NaT]))
    sn, r = describe_counts(s, {})
    assert len(r["value_counts_without_nan"].index) == 2


def test_count_summary_category():
    s = pd.Categorical(
        ["Poor", "Neutral"] + [np.nan] * 100,
        categories=["Poor", "Neutral", "Excellent"],
    )
    sn, r = describe_counts(s, {})
    assert len(r["value_counts_without_nan"].index) == 2


def test_boolean_count():
    _, results = describe_boolean_1d(
        pd.Series([{"Hello": True}, {"Hello": False}, {"Hello": True}]),
        {"hashable": True, "value_counts_without_nan": pd.Series({True: 2, False: 1})},
    )

    assert results["top"]
    assert results["freq"] == 2


@pytest.mark.sparktest
def test_boolean_count_spark(spark_session):
    sdf = spark_session.createDataFrame(
        pd.DataFrame([{"Hello": True}, {"Hello": False}, {"Hello": True}])
    )
    _, results = describe_boolean_spark_1d(
        SparkSeries(sdf), {"value_counts_without_nan": pd.Series({True: 2, False: 1})}
    )
    assert results["top"]
    assert results["freq"] == 2


@pytest.mark.sparktest
def test_describe_timestamp_spark_1d(spark_session):
    from datetime import datetime
    df = pd.DataFrame([{"Timestamp": datetime(2011, 7, 4)},
                      {"Timestamp": datetime(2022, 1, 1, 13, 57)},
                      {"Timestamp": datetime(1990, 12, 9)},
                      {"Timestamp": None},
                      {"Timestamp": np.nan},
                      {"Timestamp": datetime(1990, 12, 9)},
                      {"Timestamp": datetime(1950, 12, 9)},
                      {"Timestamp": datetime(1950, 12, 9)},
                      {"Timestamp": datetime(1950, 12, 9)},
                      {"Timestamp": datetime(1898, 1, 2)}
                      ])

    sdf = spark_session.createDataFrame(df)
    value_counts = df.value_counts().reset_index().set_index('Timestamp').squeeze()

    _, results = describe_timestamp_spark_1d(
        SparkSeries(sdf), {"n_unique": 5,
                           "value_counts_without_nan": value_counts})

    assert results["min"] == datetime(1898, 1, 2)
    assert results["max"] == datetime(2022, 1, 1, 13, 57)
    assert str(results['range']) == '45289 days 13:57:00'
