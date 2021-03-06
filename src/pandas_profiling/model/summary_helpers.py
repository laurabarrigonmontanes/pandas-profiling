import os
from collections import Counter
from datetime import datetime
from functools import partial, singledispatch
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats.stats import chisquare
from tangled_up_in_unicode import block, block_abbr, category, category_long, script

from pandas_profiling.config import config
from pandas_profiling.model.series_wrappers import SparkSeries
from pandas_profiling.model.summary_helpers_image import (
    extract_exif,
    hash_image,
    is_image_truncated,
    open_image,
)


def mad(arr):
    """Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variability of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(arr - np.median(arr)))


@singledispatch
def named_aggregate_summary(series, key: str):
    series_type = type(series)
    raise NotImplementedError(f"Function not implemented for series type {series_type}")


@named_aggregate_summary.register(pd.Series)
def _named_aggregate_summary_pandas(series: pd.Series, key: str):
    summary = {
        f"max_{key}": np.max(series),
        f"mean_{key}": np.mean(series),
        f"median_{key}": np.median(series),
        f"min_{key}": np.min(series),
    }

    return summary


@named_aggregate_summary.register(SparkSeries)
def _named_aggregate_summary_spark(series: SparkSeries, key: str):
    import pyspark.sql.functions as F

    lengths = series.dropna.select(F.length(series.name).alias("length"))

    # do not count length of nans
    numeric_results_df = (
        lengths.select(
            F.mean("length").alias("mean"),
            F.min("length").alias("min"),
            F.max("length").alias("max"),
        )
        .toPandas()
        .T
    )

    quantile_error = config["spark"]["quantile_error"].get(float)
    median = lengths.stat.approxQuantile("length", [0.5], quantile_error)[0]
    summary = {
        f"max_{key}": numeric_results_df.loc["max"][0],
        f"mean_{key}": numeric_results_df.loc["mean"][0],
        f"median_{key}": median,
        f"min_{key}": numeric_results_df.loc["min"][0],
    }

    return summary


@singledispatch
def length_summary(series, summary: dict = {}) -> dict:
    series_type = type(series)
    raise NotImplementedError(f"Function not implemented for series type {series_type}")


@length_summary.register(pd.Series)
def _length_summary_pandas(series: pd.Series, summary: dict = {}) -> dict:
    length = series.str.len()

    summary.update({"length": length})
    summary.update(named_aggregate_summary(length, "length"))

    return summary


@length_summary.register(SparkSeries)
def _length_summary_spark(series: SparkSeries, summary: dict = {}) -> dict:
    import pyspark.sql.functions as F

    length_values_sample = config["spark"]["length_values_sample"].get(int)
    if length_values_sample >= series.n_rows:
        percentage = 1.0
    else:
        percentage = length_values_sample / series.n_rows
    # do not count length of nans
    length = (
        series.dropna.select(F.length(series.name))
        .sample(percentage)
        .toPandas()
        .squeeze()
    )

    summary.update({"length": length})
    summary.update(named_aggregate_summary(series, "length"))

    return summary


def file_summary(series: pd.Series) -> dict:
    """

    Args:
        series: series to summarize

    Returns:

    """

    # Transform
    stats = series.map(lambda x: os.stat(x))

    def convert_datetime(x):
        return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")

    # Transform some more
    summary = {
        "file_size": stats.map(lambda x: x.st_size),
        "file_created_time": stats.map(lambda x: x.st_ctime).map(convert_datetime),
        "file_accessed_time": stats.map(lambda x: x.st_atime).map(convert_datetime),
        "file_modified_time": stats.map(lambda x: x.st_mtime).map(convert_datetime),
    }
    return summary


def path_summary(series: pd.Series) -> dict:
    """

    Args:
        series: series to summarize

    Returns:

    """

    # TODO: optimize using value counts
    summary = {
        "common_prefix": os.path.commonprefix(series.values.tolist())
        or "No common prefix",
        "stem_counts": series.map(lambda x: os.path.splitext(x)[0]).value_counts(),
        "suffix_counts": series.map(lambda x: os.path.splitext(x)[1]).value_counts(),
        "name_counts": series.map(lambda x: os.path.basename(x)).value_counts(),
        "parent_counts": series.map(lambda x: os.path.dirname(x)).value_counts(),
        "anchor_counts": series.map(lambda x: os.path.splitdrive(x)[0]).value_counts(),
    }

    summary["n_stem_unique"] = len(summary["stem_counts"])
    summary["n_suffix_unique"] = len(summary["suffix_counts"])
    summary["n_name_unique"] = len(summary["name_counts"])
    summary["n_parent_unique"] = len(summary["parent_counts"])
    summary["n_anchor_unique"] = len(summary["anchor_counts"])

    return summary


def url_summary(series: pd.Series) -> dict:
    """

    Args:
        series: series to summarize

    Returns:

    """
    summary = {
        "scheme_counts": series.map(lambda x: x.scheme).value_counts(),
        "netloc_counts": series.map(lambda x: x.netloc).value_counts(),
        "path_counts": series.map(lambda x: x.path).value_counts(),
        "query_counts": series.map(lambda x: x.query).value_counts(),
        "fragment_counts": series.map(lambda x: x.fragment).value_counts(),
    }

    return summary


def count_duplicate_hashes(image_descriptions: dict) -> int:
    """

    Args:
        image_descriptions:

    Returns:

    """
    counts = pd.Series(
        [x["hash"] for x in image_descriptions if "hash" in x]
    ).value_counts()
    return counts.sum() - len(counts)


def extract_exif_series(image_exifs: list) -> dict:
    """

    Args:
        image_exifs:

    Returns:

    """
    exif_keys = []
    exif_values: dict = {}

    for image_exif in image_exifs:
        # Extract key
        exif_keys.extend(list(image_exif.keys()))

        # Extract values per key
        for exif_key, exif_val in image_exif.items():
            if exif_key not in exif_values:
                exif_values[exif_key] = []

            exif_values[exif_key].append(exif_val)

    series = {"exif_keys": pd.Series(exif_keys).value_counts().to_dict()}

    for k, v in exif_values.items():
        series[k] = pd.Series(v).value_counts()

    return series


def extract_image_information(
    path: Path, exif: bool = False, hash: bool = False
) -> dict:
    """Extracts all image information per file, as opening files is slow

    Args:
        path: Path to the image
        exif: extract exif information
        hash: calculate hash (for duplicate detection)

    Returns:
        A dict containing image information
    """
    information: dict = {}
    image = open_image(path)
    information["opened"] = image is not None
    if image is not None:
        information["truncated"] = is_image_truncated(image)
        if not information["truncated"]:
            information["size"] = image.size
            if exif:
                information["exif"] = extract_exif(image)
            if hash:
                information["hash"] = hash_image(image)

    return information


def image_summary(series: pd.Series, exif: bool = False, hash: bool = False) -> dict:
    """

    Args:
        series: series to summarize
        exif: extract exif information
        hash: calculate hash (for duplicate detection)

    Returns:

    """

    image_information = series.apply(
        partial(extract_image_information, exif=exif, hash=hash)
    )
    summary = {
        "n_truncated": sum(
            [1 for x in image_information if "truncated" in x and x["truncated"]]
        ),
        "image_dimensions": pd.Series(
            [x["size"] for x in image_information if "size" in x],
            name="image_dimensions",
        ),
    }

    image_widths = summary["image_dimensions"].map(lambda x: x[0])
    summary.update(named_aggregate_summary(image_widths, "width"))
    image_heights = summary["image_dimensions"].map(lambda x: x[1])
    summary.update(named_aggregate_summary(image_heights, "height"))
    image_areas = image_widths * image_heights
    summary.update(named_aggregate_summary(image_areas, "area"))

    if hash:
        summary["n_duplicate_hash"] = count_duplicate_hashes(image_information)

    if exif:
        exif_series = extract_exif_series(
            [x["exif"] for x in image_information if "exif" in x]
        )
        summary["exif_keys_counts"] = exif_series["exif_keys"]
        summary["exif_data"] = exif_series

    return summary


@singledispatch
def get_character_counts(series: pd.Series) -> Counter:
    series_type = type(series)
    raise NotImplementedError(f"Function not implemented for series type {series_type}")


@get_character_counts.register(pd.Series)
def _get_character_counts_pandas(series: pd.Series) -> Counter:
    """Function to return the character counts

    Args:
        series: the Series to process

    Returns:
        A dict with character counts
    """
    return Counter(series.str.cat())


@get_character_counts.register(SparkSeries)
def _get_character_counts_spark(series: SparkSeries) -> Counter:
    """Function to return the character counts

    Args:
        series: the Series to process

    Returns:
        A dict with character counts
    """
    import pyspark.sql.functions as F

    # this function is optimised to split all characters and explode the characters and then groupby the characters
    # because the number of characters is limited, the return dataset is small -> can return everything to pandas
    df = (
        series.dropna.select(F.explode(F.split(F.col(series.name), "")))
        .groupby("col")
        .count()
        .toPandas()
    )

    # standardise return as Counter object
    my_dict = Counter({x[0]: x[1] for x in zip(df["col"].values, df["count"].values())})
    return my_dict


def counter_to_series(counter: Counter) -> pd.Series:
    if not counter:
        return pd.Series([], dtype=object)

    counter_as_tuples = counter.most_common()
    items, counts = zip(*counter_as_tuples)
    return pd.Series(counts, index=items)


def unicode_summary(series) -> dict:
    # Unicode Character Summaries (category and script name)

    # this is the function that properly computes the character counts based on type
    character_counts = get_character_counts(series)

    character_counts_series = counter_to_series(character_counts)

    char_to_block = {key: block(key) for key in character_counts.keys()}
    char_to_category_short = {key: category(key) for key in character_counts.keys()}
    char_to_script = {key: script(key) for key in character_counts.keys()}

    summary = {
        "n_characters": len(character_counts_series),
        "character_counts": character_counts_series,
        "category_alias_values": {
            key: category_long(value) for key, value in char_to_category_short.items()
        },
        "block_alias_values": {
            key: block_abbr(value) for key, value in char_to_block.items()
        },
    }
    # Retrieve original distribution
    block_alias_counts: Counter = Counter()
    per_block_char_counts: dict = {
        k: Counter() for k in summary["block_alias_values"].values()
    }
    for char, n_char in character_counts.items():
        block_name = summary["block_alias_values"][char]
        block_alias_counts[block_name] += int(n_char)
        per_block_char_counts[block_name][char] = n_char
    summary["block_alias_counts"] = counter_to_series(block_alias_counts)
    summary["block_alias_char_counts"] = {
        k: counter_to_series(v) for k, v in per_block_char_counts.items()
    }

    script_counts: Counter = Counter()
    per_script_char_counts: dict = {k: Counter() for k in char_to_script.values()}
    for char, n_char in character_counts.items():
        script_name = char_to_script[char]
        script_counts[script_name] += int(n_char)
        per_script_char_counts[script_name][char] = n_char
    summary["script_counts"] = counter_to_series(script_counts)
    summary["script_char_counts"] = {
        k: counter_to_series(v) for k, v in per_script_char_counts.items()
    }

    category_alias_counts: Counter = Counter()
    per_category_alias_char_counts: dict = {
        k: Counter() for k in summary["category_alias_values"].values()
    }
    for char, n_char in character_counts.items():
        category_alias_name = summary["category_alias_values"][char]
        category_alias_counts[category_alias_name] += int(n_char)
        per_category_alias_char_counts[category_alias_name][char] += n_char
    summary["category_alias_counts"] = counter_to_series(category_alias_counts)
    summary["category_alias_char_counts"] = {
        k: counter_to_series(v) for k, v in per_category_alias_char_counts.items()
    }

    # Unique counts
    summary["n_category"] = len(summary["category_alias_counts"])
    summary["n_scripts"] = len(summary["script_counts"])
    summary["n_block_alias"] = len(summary["block_alias_counts"])
    if len(summary["category_alias_counts"]) > 0:
        summary["category_alias_counts"].index = summary[
            "category_alias_counts"
        ].index.str.replace("_", " ")

    return summary


def histogram_compute(finite_values, n_unique, name="histogram", weights=None):
    stats = {}
    bins = config["plot"]["histogram"]["bins"].get(int)
    bins = "auto" if bins == 0 else min(bins, n_unique)

    stats[name] = np.histogram(finite_values, bins=bins, weights=weights)

    max_bins = config["plot"]["histogram"]["max_bins"].get(int)
    if bins == "auto" and len(stats[name][1]) > max_bins:
        stats[name] = np.histogram(finite_values, bins=max_bins, weights=None)

    return stats


def histogram_compute_spark(sparkseries, bins, n_unique, name="histogram"):
    stats = {}
    config_bins = config["spark"]["histogram_bins"].get(int)
    bins = bins if config_bins == 0 else min(bins, n_unique)

    spark_histogram = (
        sparkseries.dropna.select(sparkseries.name)
        .rdd.flatMap(lambda x: x)
        .histogram(bins)
    )

    # Loading the Computed Histogram into a Pandas Dataframe for plotting
    computed_bins = [i[0] for i in spark_histogram]
    weights = [i[1] for i in spark_histogram]
    stats[name] = np.histogram(computed_bins, bins=bins, weights=weights)
    return stats


def chi_square(values=None, histogram=None):
    if histogram is None:
        histogram, _ = np.histogram(values, bins="auto")
    return dict(chisquare(histogram)._asdict())


def chi_square_spark(series):
    """
    currently unused, this is a function to compute chisquare using spark, but it slows down compute a lot.

    Args:
        series:

    Returns:

    """
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.mllib.stat import Statistics

    vector_name = series.name + "assembled"
    vecAssembler = StringIndexer(inputCol=series.name, outputCol=vector_name)
    vec = vecAssembler.fit(series.dropna).transform(series.dropna)
    vec = vec.select(vector_name)
    vec = vec.collect()
    results = Statistics.chiSqTest(vec)
    return {"statistic": results.statistic, "pvalue": results.pValue}
