from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

class ProfitGenerator(Transformer):
  """
  A custom transformer which generates a target column for regression. The target column indicates the
  profits that can be made from players in two years. The profit is based on the value of players in 2019
  and in 2021.
  """

  def __init__(self):
    super(ProfitGenerator, self).__init__()

  def _transform(self, df: DataFrame) -> DataFrame:
    # creating new column to indicate if profits has been made in two years for players
    df = df.withColumn('Gain in Two Years',F.col('Value_2021(M)') - F.col('Value_2019(M)'))
    return df


class GrowthGenerator(Transformer):
  """
  A custom transformer which generates targets for classification. The target column indicates a player's
  growth level as high growth, low growth, mid growth and no growth. These growth levels indicates the
  potential for players to grow in 2 years time.
  """

  def __init__(self):
    super(GrowthGenerator, self).__init__()

  def _transform(self, df: DataFrame) -> DataFrame:

    # creating a function to categorise players on growth level
    def player_growth(potential_19, potential_21):
      potential_difference = float(potential_21) - float(potential_19)
      #if potential_21.isNull():
      #  return 'No_Growth'
      if potential_difference > 5:
        return 'High_Growth'
      elif ((potential_difference > 2) & (potential_difference <= 5)):
        return 'Mid_Growth'
      elif ((potential_difference <= 2) & (potential_difference > 0)):
        return 'Low_Growth'
      else:
        return 'No_Growth'

    # converting the above function to a pyspark user defined function
    udf_player_growth = F.udf(lambda y, z:player_growth(y,z),StringType())
    #udf_player_growth = F.udf(player_growth, StringType())
    # creating a column named growth level based on the above function
    df = df.withColumn('Growth_Level',udf_player_growth(df['Potential_2019'], df['Potential_2021']))
    return df