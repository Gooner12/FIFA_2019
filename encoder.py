from pyspark.ml import Transformer
from pyspark.sql import DataFrame

# defining a custom transformer to perform label encoding for players' position group
class LabelEncoder(Transformer):
    """
    A custom Transformer which creates a target column based on the encoding of position group, such as
    if position group is DEF, it is encoded as 1, 0 for FWD and 3 for MID
    """

    def __init__(self):
        super(LabelEncoder, self).__init__()
        #self.position_group = position_group

    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.withColumn('Target',when(col('Position_Group') == 'DEF', 1).when(col('Position_Group')== 'FWD', 0)\
      .otherwise(2))
        return df


# defining a custom transformer to perform label encoding for player growth
class GrowthEncoder(Transformer):
    """
    A custom Transformer which creates a target column based on the encoding of players' growth level, such as
    if growth level is no growth, it is encoded as 0, 1 for mid growth and 3 for high growth
    """

    def __init__(self):
        super(GrowthEncoder, self).__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.withColumn('Target_Growth',when(col('Growth_Level') == 'No_Growth', 0).when(col('Growth_Level')== 'Mid_Growth', 1)\
      .otherwise(2))
        return df