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

    def __init__(self):
        self.variation_list = []
        super(ValueImputer, self).__init__()

    def variation_age(self, category, age1, age2=None):

        # calculating the percentage change in players' values
        df_diff = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNotNull()))
        df_diff = df_diff.withColumn('Variation', \
                                     (df_diff['Value_2021(M)'] - df_diff['Value_2019(M)']) / df_diff[
                                         'Value_2019(M)'])

        # finding the average variation level for the age group
        if(category == 'under'):
            if(age2 is None):
                age_group = df_diff.filter((F.col('Age') <= age1)).select(F.avg('Variation').alias('Avg')).collect()[0]
            else:
                raise ValueError('Age2 can be used only in between category')

        elif(category == 'between'):
            if(age2 is not None):
                age_group = df_diff.filter((F.col('Age') > age1) & (F.col('Age') <= age2)).select(
                        F.avg('Variation').alias('Avg')).collect()[
                        0]

            else:
                raise ValueError('Age2 is required for between category')

        elif(category == 'over'):
            if(age2 is None):
                age_group = df_diff.filter((F.col('Age') > age1)).select(F.avg('Variation').alias('Avg')).collect()[0]
                if (age_group['Avg'] is None):
                    age_group_float = 0
            else:
                raise ValueError('Age2 can be used only in between category')

        age_group_float = age_group['Avg']
        self.variation_list = self.variation_list + [age_group_float]
        return age_group_float


    def variation_calculator(self, df):

        # finding the average variation level for different age groups
        self.variation_age('under', 20)
        self.variation_age('between', 20, 25)
        self.variation_age('between', 25, 30)
        self.variation_age('between', 30, 35)
        self.variation_age('between', 35, 40)
        self.variation_age('over', 40)
        return self.variation_list

    def imputer(self, category, year,  age1, age2=None):
        if(year == 2021):
            true_operator = operator.mul
            # selecting the portion of the dataframe that has missing values in 2021 but not in 2019 for same records
            true_df = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNull()))
        elif(year == 2019):
            true_operator = operator.floordiv
            # selecting a portion of the dataframe where missing values are present in 2019 but not in 2021
            true_df = df.filter((F.col('Value_2019(M)').isNull()) & (F.col('Value_2021(M)').isNotNull()))

        if(category == 'under'):
            if(age2 is None):
                # imputing the missing values in Values_2021(M) for different age groups based on the value growth seen for respective groups
                true_df = true_df.withColumn('Value_2021(M)', F.when(F.col('Age') <= age1, \
                                                                                       F.round(true_operator(
                                                                                           F.col('Value_2019(M)'), (1 +
                                                                                                                     self.variation_list[
                                                                                                                         0])),
                                                                                           3)). \
                                                               otherwise(F.col('Value_2021(M)')))
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
                true_df  = true_df .withColumn('Value_2021(M)', \
                                                               F.when((F.col('Age') > age1) & (F.col('Age') <= age2),
                                                                      F.round(true_operator(F.col('Value_2019(M)'), (
                                                                                  1 + self.variation_list[index])), 3)). \
                                                               otherwise(F.col('Value_2021(M)')))
            else:
                raise ValueError('Age2 is required for between category')

        elif(category == 'over'):
            if(age2 is None):
                true_df  = true_df.withColumn('Value_2021(M)', \
                                                               F.when(F.col('Age') > age1, \
                                                                      F.round(true_operator(F.col('Value_2019(M)'), (
                                                                                  1 + self.variation_list[5])),3)). \
                                                               otherwise(F.col('Value_2021(M)')))
            else:
                raise ValueError('Age2 can be used only in between category')
        return df_2019_not_null



    def _transform(self, df: DataFrame) -> DataFrame:
        self.variation_calculator(df)
        df_2019_not_null = self.imputer('under', year=2021, age1=20)
        df_2019_not_null = self.imputer('between', year=2021, age1=20, age2=25)
        df_2019_not_null = self.imputer('between', year=2021, age1=25, age2=30)
        df_2019_not_null = self.imputer('between', year=2021, age1=30, age2=35)
        df_2019_not_null = self.imputer('between', year=2021, age1=35, age2=40)
        df_2019_not_null = self.imputer('over', year=2021, age1=40)

        df_2021_not_null = self.imputer('under', year=2019, age1=20)
        df_2021_not_null = self.imputer('between', year=2019, age1=20, age2=25)
        df_2021_not_null = self.imputer('between', year=2019, age1=25, age2=30)
        df_2021_not_null = self.imputer('between', year=2019, age1=30, age2=35)
        df_2021_not_null = self.imputer('between', year=2019, age1=35, age2=40)
        df_2021_not_null = self.imputer('over', year=2019, age1=40)

        # selecting a portion of the dataframe where players values are missing in both 2019 and 2021
        df_both_null = df.filter(((F.col('Value_2019(M)').isNull()) & (F.col('Value_2021(M)').isNull())))

        # selecting a portion of the dataframe where no missing values are present in both 2019 and 2021 for players
        df_both_not_null = df.filter((F.col('Value_2019(M)').isNotNull()) & (F.col('Value_2021(M)').isNotNull()))
        final_df = df_both_not_null.union(df_2019_not_null).union(df_2021_not_null).union(df_both_null)

        regression_df = final_df.na.drop(subset=['Value_2019(M)'])

        return regression_df