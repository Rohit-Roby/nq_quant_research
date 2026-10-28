
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, TimestampType, DoubleType
from quant_research.transform.validation import convert_timestamp, validate_ohlcv, validate_duplicates

def test_validate_ohlcv(spark):
    
    records = [
            ("NQ", "2026-10-26 14:00:00", 25000.0, 25020.0, 24990.0, 25010.0, 1500),
            ("NQ", "2026-10-26 14:15:00", 25010.0, 25035.0, 25005.0, 25030.0, 1800),
            ("NQ", "2026-10-26 14:30:00", 25030.0, 25040.0, 25010.0, 25015.0, 1200),
            ("NQ", "2026-10-26 14:45:00", 25000.0, 24980.0, 24990.0, 25010.0, 1000),
            ("NQ", "2026-10-26 15:00:00", 25000.0, None, 24990.0, 25010.0, 1500)]
    
    schema = StructType([
            StructField("symbol",StringType(), nullable=False),
            StructField("timestamp", TimestampType(), nullable=False),
            StructField("open", DoubleType(), nullable= False),
            StructField("high", DoubleType(), nullable= True),
            StructField("low", DoubleType(), nullable= False),
            StructField("close", DoubleType(), nullable= False),
            StructField("volume", IntegerType(), nullable= False)       
    
        ])
    
    new_record = []
    for record in records:
            record = convert_timestamp(record)
            new_record.append(record)


    df = spark.createDataFrame(new_record, schema=schema)
    df.printSchema()

    valid_df, invalid_df = validate_ohlcv(df)
    duplicates_df = validate_duplicates(df)

    assert valid_df.count() == 3
    assert invalid_df.count() == 2

 