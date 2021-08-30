from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import operator


class ValueManipulator(Transformer):
    """
    A custom transformer which converts the unit of players' values in 2021 to millions and changes the names
    of columns in a dataframe to ease the differentiation of common columns, such as potential, overall and
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

    def __init__(self, variation_list):
        # populating the list from the variations obatained from the test data
        self.variation_list = variation_list
        super(ValueImputer, self).__init__()

    def imputer(self, df, category, year,  age1, age2=None):
        if(year == 2021):
            true_operator = operator.mul
            # selecting the portion of the dataframe that has missing values in 2021 but not in 2019 for same records
            true_df = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNull()))
            col_name1 = 'Value_2019(M)'
            col_name2 = 'Value_2021(M)'
        elif(year == 2019):
            true_operator = operator.truediv
            # selecting a portion of the dataframe where missing values are present in 2019 but not in 2021
            true_df = df.filter((F.col('Value_2019(M)').isNull()) & (F.col('Value_2021(M)').isNotNull()))
            col_name1 = 'Value_2021(M)'
            col_name2 = 'Value_2019(M)'

        if(category == 'under'):
            if(age2 is None):
                # imputing the missing values in the required column for different age groups based on the value growth seen for respective groups
                true_df = true_df.withColumn(col_name2, F.when(F.col('Age') <= age1, \
                                                                                       F.round(true_operator(
                                                                                           F.col(col_name1), (1 +
                                                                                                                     self.variation_list[
                                                                                                                         0])),
                                                                                           3)). \
                                                               otherwise(F.col(col_name2)))
                true_df = true_df.filter(F.col('Age') <= age1)
            else:
                raise ValueError('Age2 can be used only in between category')

        elif(category == 'between'):
            if(age2 == 25):
                index = 1
            elif(age2 == 30):
                index = 2
            elif(age2 == 35):
                index = 3
            elif(age2 == 40):
                index = 4
            if(age2 is not None):
                true_df  = true_df.withColumn(col_name2, \
                                                               F.when((F.col('Age') > age1) & (F.col('Age') <= age2),
                                                                      F.round(true_operator(F.col(col_name1), (
                                                                                  1 + self.variation_list[index])), 3)). \
                                                               otherwise(F.col(col_name2)))
                true_df = true_df.filter((F.col('Age') > age1) & (F.col('Age') <= age2))
            else:
                raise ValueError('Age2 is required for between category')

        elif(category == 'over'):
            if(age2 is None):
                true_df  = true_df.withColumn(col_name2, \
                                                               F.when(F.col('Age') > age1, \
                                                                      F.round(true_operator(F.col(col_name1), (
                                                                                  1 + self.variation_list[5])),3)). \
                                                               otherwise(F.col(col_name2)))
                true_df = true_df.filter(F.col('Age') > age1)
            else:
                raise ValueError('Age2 can be used only in between category')
        return true_df



    def _transform(self, df: DataFrame) -> DataFrame:
        df_2019_not_null_a = self.imputer(df, 'under', year=2021, age1=20)
        df_2019_not_null_b = self.imputer(df, 'between', year=2021, age1=20, age2=25)
        df_2019_not_null_c = self.imputer(df, 'between', year=2021, age1=25, age2=30)
        df_2019_not_null_d = self.imputer(df, 'between', year=2021, age1=30, age2=35)
        df_2019_not_null_e = self.imputer(df, 'between', year=2021, age1=35, age2=40)
        df_2019_not_null_f = self.imputer(df, 'over', year=2021, age1=40)
        df_2019_not_null = df_2019_not_null_a.union(df_2019_not_null_b).union(df_2019_not_null_c).\
        union(df_2019_not_null_d).union(df_2019_not_null_e).union(df_2019_not_null_f)

        df_2021_not_null_a = self.imputer(df, 'under', year=2019, age1=20)
        df_2021_not_null_b = self.imputer(df, 'between', year=2019, age1=20, age2=25)
        df_2021_not_null_c = self.imputer(df, 'between', year=2019, age1=25, age2=30)
        df_2021_not_null_d = self.imputer(df, 'between', year=2019, age1=30, age2=35)
        df_2021_not_null_e = self.imputer(df, 'between', year=2019, age1=35, age2=40)
        df_2021_not_null_f = self.imputer(df, 'over', year=2019, age1=40)
        df_2021_not_null = df_2021_not_null_a.union(df_2021_not_null_b).union(df_2021_not_null_c).\
        union(df_2021_not_null_d).union(df_2021_not_null_e).union(df_2021_not_null_f)

        # selecting a portion of the dataframe where players values are missing in both 2019 and 2021
        df_both_null = df.filter(((F.col('Value_2019(M)').isNull()) & (F.col('Value_2021(M)').isNull())))

        # selecting a portion of the dataframe where no missing values are present in both 2019 and 2021 for players
        df_both_not_null = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNotNull()))

        # we are not including records containing both null values, so we do not include df_both_null in the final dataframe
        final_df = df_both_not_null.union(df_2019_not_null).union(df_2021_not_null)

        regression_df = final_df

        return regression_df