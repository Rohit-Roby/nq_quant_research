from pyspark.sql import SparkSession, Row
from datetime import datetime, date
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, TimestampType, DoubleType
from pyspark.sql import functions as F
spark = SparkSession.builder.master("local[*]").appName("NQQuantResearch").getOrCreate()


def convert_timestamp(record):
    
    timestamp_format = '%Y-%m-%d %H:%M:%S'

    timestamp = datetime.strptime(record[1], timestamp_format)
    new_record = record[:1] + (timestamp,) + record[2:]

    return new_record

def validate_ohlcv(df):
    valid_condition = (
        (F.col("high") >= F.col("low")) &
        (F.col("high") >= F.col("close")) &
        (F.col("low") <= F.col("open")) &
        (F.col("low") <= F.col("close")) &
        (F.col("high") >= F.col("open")) &
        (F.col("volume") >= 0) &
        (F.col("open").isNotNull()) &
        (F.col("high").isNotNull()) &
        (F.col("low").isNotNull()) &
        (F.col("close").isNotNull()) &
        (F.col("volume").isNotNull()) 
    )
    valid_df = df.filter(valid_condition)
    invalid_df = df.filter(~valid_condition)

    return valid_df, invalid_df

def validate_nulls(df):

    valid_null_condition = (
        (F.col("open").isNotNull()) &
        (F.col("high").isNotNull()) &
        (F.col("low").isNotNull()) &
        (F.col("close").isNotNull()) &
        (F.col("volume").isNotNull()) 
    )
    valid_df = df.filter(valid_null_condition)
    invalid_df = df.filter(~valid_null_condition)

    return valid_df, invalid_df

def validate_duplicates(df):
    
    duplicates_df = (
        df.groupBy("symbol", "timestamp")
        .count()
        .filter(F.col("count")>1)
    )
    
    return duplicates_df

def find_confliction_duplicates(df):
    unique_versions = df.distinct()
    
    conflicts_df = (
        unique_versions.groupBy("symbol", "timestamp")
        .count()
        .filter(F.col("count") > 1)
    )
    return conflicts_df

def quarantine_conflicts(df, conflicts_df):
    clean_df = df.join(
        conflicts_df.select("symbol","timestamp"),
        on=["symbol", "timestamp"],
        how="left_anti"
    )
    quarantine_df = df.join(
        conflicts_df.select("symbol","timestamp"),
        on=["symbol","timestamp"],
        how="inner"
    )

    return clean_df, quarantine_df

def process_market_data(df):

    conflicts_df = find_confliction_duplicates(df)

    remaining_df, conflict_quarantine_df = quarantine_conflicts(df, conflicts_df)

    valid_df, invalid_ohlcv_df = validate_ohlcv(remaining_df)

    # duplicates_df =  validate_duplicates(valid_df)
    silver_df = valid_df.dropDuplicates()

    return silver_df, invalid_ohlcv_df, conflict_quarantine_df


try:
    records = [
        ("NQ", "2026-10-26 14:00:00", 25000.0, 25020.0, 24990.0, 25010.0, 1500),
        ("NQ", "2026-10-26 14:15:00", 25010.0, 25035.0, 25005.0, 25030.0, 1800),
        ("NQ", "2026-10-26 14:30:00", 25030.0, 25040.0, 25010.0, 25015.0, 1200),
        ("NQ", "2026-10-26 14:45:00", 25000.0, 24980.0, 24990.0, 25010.0, 1000),
        ("NQ", "2026-10-26 14:45:00", 25000.0, 24980.0, 24990.0, 25010.0, 1000),
        ("NQ", "2026-10-26 14:45:00", 25000.0, 24950.0, 24970.0, 25000.0, 1200),
        ("NQ", "2026-10-26 15:00:00", 25000.0, None, 24990.0, 25010.0, 1500)]

    new_record=[]
    for record in records:
        record = convert_timestamp(record)
        new_record.append(record)


    schema = StructType([
        StructField("symbol",StringType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("open", DoubleType(), nullable= False),
        StructField("high", DoubleType(), nullable= True),
        StructField("low", DoubleType(), nullable= False),
        StructField("close", DoubleType(), nullable= False),
        StructField("volume", IntegerType(), nullable= False)
    
    ])

    spark.sparkContext.setLogLevel("ERROR")

    df = spark.createDataFrame(new_record, schema=schema)
    df.printSchema()

    silver_df, invalid_ohlcv_df, conflict_quarantine_df = process_market_data(df)

    print("SILVER STAGE DATA : ")
    silver_df.show()
    print("INVALID OHLCV DATA : ")
    invalid_ohlcv_df.show()
    print("CONFLICT QUARANTINE DATA : ")
    conflict_quarantine_df.show()


finally:
    spark.stop()
