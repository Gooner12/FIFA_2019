# check
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class ValueManipulator(Transformer):
    """
    A custom transformer which converts the unit of players' values in 2021 to millions and changes the names
    of columns in a dataframe to ease the differencing of common columns, such as potential, overall and
    value, based on the year in which the values are recorded.
    """

    def __init__(self):
        super(ValueManipulator, self).__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        # changing the units of player's value in 2021 to millions
        df = df.withColumn('value_eur', F.col('value_eur') / 1000000)

        # defining a function to rename the columns in dataframe
        def rename_col(df, col_names):
            for old_name, new_name in col_names.items():
                df = df.withColumnRenamed(old_name, new_name)
            return df

            # storing the dataframe with renamed columns based on years

        names = [('Overall', 'Overall_2019'), ('Potential', 'Potential_2019'), ('overall', 'Overall_2021'), \
                 ('potential', 'Potential_2021'), ('value(M)', 'Value_2019(M)'), ('value_eur', 'Value_2021(M)')]
        df = rename_col(df, dict(names))
        return df


class UnifyValue(Transformer):
    """
    A custom transformer which selects non-goalkeepers, imputes null values present in potential for 2021
    and converts the values of players which are given in thousands(K) instead of millions to millions to
    standardise the players values in 2019.
    """

    def __init__(self):
        super(UnifyValue, self).__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        # removing goalkeepers from the dataframe
        df = df.filter(df['Position'] != 'GK')

        # replacing null values in new potential column with zero as we indicate these players as no future growth players later
        # df = df.withColumn('Potential_2021',F.when(F.col('Potential_2021').isNull(),0).otherwise(F.col('Potential_2021')))
        df = df.fillna(0, subset=['Potential_2021'])

        # standardising the values of players to millions as there are players whose values are indicated in K
        df_k = df.filter(df['Value_2019(M)'].contains('K'))
        df_non_k = df.filter(~df['Value_2019(M)'].contains('K'))
        df_null = df.filter(df['Value_2019(M)'].isNull())
        df_k = df_k.withColumn('Value_2019(M)', F.when(F.col('Value_2019(M)').contains('K'), \
                                                       F.regexp_replace(F.col('Value_2019(M)'), 'K', '')).otherwise(
            F.col('Value_2019(M)')))
        df_k = df_k.withColumn('Value_2019(M)', F.col('Value_2019(M)') / 1000)

        # combining the dataframes to get one complete dataframe
        df = df_non_k.union(df_k).union(df_null)

        # changing the datatype of Value_2019(M) column to float
        df = df.withColumn('Value_2019(M)', df['Value_2019(M)'].cast('float'))

        return df


class ValueImputer(Transformer):
    """
    A custom transformer which imputes the missing values present in players values for both 2019 and 2021.
    If there are values for 2019 but for same rows values are missing for 2021, then based on the growth for
    a specific age group, missing values for 2021 are imputed according to growth on 2019 figures. Similarly,
    if values are missing for 2019 but present in 2021, based on the growth level, using 2021 values, missing
    values for 2019 are imputed. All these operations are performed in segments which are finally concatenated.
    """

    def __init__(self):
        super(ValueImputer, self).__init__()

    def _transform(self, df: DataFrame) -> DataFrame:
        # calculating the percentage change in players' values
        df_diff = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNotNull()))
        df_diff = df_diff.withColumn('Variation', \
                                     (df_diff['Value_2021(M)'] - df_diff['Value_2019(M)']) / df_diff[
                                         'Value_2019(M)'])

        # finding the average variation level for different age groups
        age_under_21 = df_diff.filter((F.col('Age') <= 20)).select(F.avg('Variation').alias('Avg')).collect()[0]
        age_under_21_float = age_under_21['Avg']

        age_21_to_25 = \
        df_diff.filter((F.col('Age') > 20) & (F.col('Age') <= 25)).select(F.avg('Variation').alias('Avg')).collect()[
            0]
        age_21_to_25_float = age_21_to_25['Avg']

        age_26_to_30 = \
        df_diff.filter((F.col('Age') > 25) & (F.col('Age') <= 30)).select(F.avg('Variation').alias('Avg')).collect()[
            0]
        age_26_to_30_float = age_26_to_30['Avg']

        age_31_to_35 = \
        df_diff.filter((F.col('Age') > 30) & (F.col('Age') <= 35)).select(F.avg('Variation').alias('Avg')).collect()[
            0]
        age_31_to_35_float = age_31_to_35['Avg']

        age_36_to_40 = \
        df_diff.filter((F.col('Age') > 35) & (F.col('Age') <= 40)).select(F.avg('Variation').alias('Avg')).collect()[
            0]
        age_36_to_40_float = age_36_to_40['Avg']

        age_greater_40 = df_diff.filter((F.col('Age') > 40)).select(F.avg('Variation').alias('Avg')).collect()[0]
        age_greater_40_float = age_greater_40['Avg']
        if (age_greater_40_float is None):
            age_greater_40_float = 0

        # selecting the portion of the dataframe that has missing values in 2021 but not in 2019 for same records
        df_2019_not_null = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNull()))

        # imputing the missing values in Values_2021(M) for different age groups based on the value growth seen for respective groups
        df_2019_not_null = df_2019_not_null.withColumn('Value_2021(M)', F.when(F.col('Age') <= 20, \
                                                                               F.round(
                                                                                   F.col('Value_2019(M)') * (1 + age_under_21_float),
                                                                                   3)). \
                                                       otherwise(F.col('Value_2021(M)')))

        df_2019_not_null = df_2019_not_null.withColumn('Value_2021(M)', \
                                                       F.when((F.col('Age') > 20) & (F.col('Age') <= 25),
                                                              F.round(F.col('Value_2019(M)') * (1 + age_21_to_25_float), 3)). \
                                                       otherwise(F.col('Value_2021(M)')))

        df_2019_not_null = df_2019_not_null.withColumn('Value_2021(M)', \
                                                       F.when((F.col('Age') > 25) & (F.col('Age') <= 30),
                                                              F.round(F.col('Value_2019(M)') * (1 + age_26_to_30_float), 3)). \
                                                       otherwise(F.col('Value_2021(M)')))

        df_2019_not_null = df_2019_not_null.withColumn('Value_2021(M)', \
                                                       F.when((F.col('Age') > 30) & (F.col('Age') <= 35),
                                                              F.round(F.col('Value_2019(M)') * (1 + age_31_to_35_float), 3)). \
                                                       otherwise(F.col('Value_2021(M)')))

        df_2019_not_null = df_2019_not_null.withColumn('Value_2021(M)', \
                                                       F.when((F.col('Age') > 35) & (F.col('Age') <= 40),
                                                              F.round(F.col('Value_2019(M)') * (1 + age_36_to_40_float), 3)). \
                                                       otherwise(F.col('Value_2021(M)')))

        df_2019_not_null = df_2019_not_null.withColumn('Value_2021(M)', \
                                                       F.when(F.col('Age') > 40, \
                                                              F.round(F.col('Value_2019(M)') * (1+ age_greater_40_float))). \
                                                       otherwise(F.col('Value_2021(M)')))

        # selecting a portion of the dataframe where players values are missing in both 2019 and 2021
        df_both_null = df.filter(((F.col('Value_2019(M)').isNull()) & (F.col('Value_2021(M)').isNull())))

        # selecting a portion of the dataframe where missing values are present in 2019 but not in 2021
        df_2021_not_null = df.filter((F.col('Value_2019(M)').isNull()) & (F.col('Value_2021(M)').isNotNull()))

        # imputing the missing values in Values_2019(M) for different age groups based on the value growth seen for respective groups
        df_2021_not_null = df_2021_not_null.withColumn('Value_2019(M)', F.when(F.col('Age') <= 20, \
                                                                               F.round(
                                                                                   F.col('Value_2021(M)') / (1 + age_under_21_float),
                                                                                   3)). \
                                                       otherwise(F.col('Value_2019(M)')))

        df_2021_not_null = df_2021_not_null.withColumn('Value_2019(M)', \
                                                       F.when((F.col('Age') > 20) & (F.col('Age') <= 25), \
                                                              F.round(F.col('Value_2021(M)') / (1 + age_21_to_25_float), 3)). \
                                                       otherwise(F.col('Value_2019(M)')))

        df_2021_not_null = df_2021_not_null.withColumn('Value_2019(M)', \
                                                       F.when((F.col('Age') > 25) & (F.col('Age') <= 30), \
                                                              F.round(F.col('Value_2021(M)') / (1 + age_26_to_30_float), 3)). \
                                                       otherwise(F.col('Value_2019(M)')))

        df_2021_not_null = df_2021_not_null.withColumn('Value_2019(M)', \
                                                       F.when((F.col('Age') > 30) & (F.col('Age') <= 35), \
                                                              F.round(F.col('Value_2021(M)') / (1 + age_31_to_35_float), 3)). \
                                                       otherwise(F.col('Value_2019(M)')))

        df_2021_not_null = df_2021_not_null.withColumn('Value_2019(M)', \
                                                       F.when((F.col('Age') > 35) & (F.col('Age') <= 40), \
                                                              F.round(F.col('Value_2021(M)') / (1 + age_36_to_40_float), 3)). \
                                                       otherwise(F.col('Value_2019(M)')))

        df_2021_not_null = df_2021_not_null.withColumn('Value_2019(M)', F.when(F.col('Age') > 40, \
                                                                               F.round(F.col('Value_2021(M)') / (1+age_greater_40_float))). \
                                                       otherwise(F.col('Value_2019(M)')))

        # selecting a portion of the dataframe where no missing values are present in both 2019 and 2021 for players
        df_both_not_null = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNotNull()))
        final_df = df_both_not_null.union(df_2019_not_null).union(df_2021_not_null).union(df_both_null)

        regression_df = final_df.na.drop(subset=['Value_2019(M)'])

        return regression_df