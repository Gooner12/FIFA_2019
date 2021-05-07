from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql.functions import when, col

# defining a custom transformer to perform label encoding for players' position group
class LabelEncoder(Transformer):
    """
    A custom Transformer which creates a target column based on the type of columns.
    For column, position group, the encoding is performed in a way that if position group is DEF,
    it is encoded as 1, 0 for FWD and 3 for MID. For column, Growth_Level, the encoding is done
    in a way that if growth level is no growth, it is encoded as 0, 1 for mid growth and 3 for
    high growth.
    """

    def __init__(self):
        super(LabelEncoder, self).__init__()
        #self.position_group = position_group

    def _transform(self, df: DataFrame) -> DataFrame:
        if('Position_Group' in df.columns):
            df = df.withColumn('Target', when(col('Position_Group') == 'DEF', 1).when(col('Position_Group')== 'FWD', 0)\
          .otherwise(2))
        elif('Growth_Level' in df.columns):
            df = df.withColumn('Target', when(col('Growth_Level') == 'No_Growth', 0).when(col('Growth_Level') == 'Low_Growth', 1) \
                               .when(col('Growth_Level') == 'Mid_Growth', 2).otherwise(3))
        return df


