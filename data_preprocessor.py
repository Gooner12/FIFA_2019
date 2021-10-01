from pyspark.ml import Transformer
from pyspark.ml.pipeline import Estimator
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

class ValueImputer(Estimator):

    def __init__(self, col1, col2, num):
        self.variation_list = []
        self.col1 = col1
        self.col2 = col2
        self.num = num
        self.age_1 = [20,20,25,30,35,40]
        self.age_2 = [None,25,30,35,40,None]
        self.categories = ['under', 'between', 'between', 'between', 'between', 'over']
        super(ValueImputer, self).__init__()

    def variation_age(self, df, category, col1, col2, age1, age2):

        # calculating the percentage change in players' values
        df_diff = df.filter((F.col(col1).isNotNull()) & (F.col(col2).isNotNull()))
        df_diff = df_diff.withColumn('Variation', \
                                     (df_diff[col2] - df_diff[col1]) / df_diff[col1])

        # finding the average variation level for the age group
        if(category == 'under'):
          age_group = df_diff.filter((F.col('Age') <= age1)).select(F.avg('Variation').alias('Avg')).collect()[0]

        elif(category == 'between'):
          age_group = df_diff.filter((F.col('Age') > age1) & (F.col('Age') <= age2)).select(
                        F.avg('Variation').alias('Avg')).collect()[
                        0]

        elif(category == 'over'):
          age_group = df_diff.filter((F.col('Age') > age1)).select(F.avg('Variation').alias('Avg')).collect()[0]

        age_group_float = age_group['Avg']
        if (age_group_float is None):
            age_group_float = 0
        self.variation_list = self.variation_list + [age_group_float]
        #return age_group_float

    def variation_calculator(self, df):
        # finding the average variation level for different age groups
        for i, j, k in zip(self.age_1, self.age_2, self.categories):
          self.variation_age(df, k, self.col1, self.col2, i, j)

    def _fit(self, df:DataFrame):
        self.variation_calculator(df)
        return ValueTransformer(self.col1, self.col2, self.num, self.variation_list)


class ValueTransformer(Transformer):
    """
    A custom transformer which imputes the missing values present in players values for both 2019 and 2021.
    If there are values for 2019 but for same rows values are missing for 2021, then based on the growth for
    a specific age group, missing values for 2021 are imputed according to growth on 2019 figures. Similarly,
    if values are missing for 2019 but present in 2021, based on the growth level, using 2021 values, missing
    values for 2019 are imputed. All these operations are performed in segments which are finally concatenated.
    """

    def __init__(self, col1, col2, num, variation_list):
        self.variation_list = variation_list
        self.col1 = col1
        self.col2 = col2
        self.num = num
        self.age_1 = [20,20,25,30,35,40]
        self.age_2 = [None,25,30,35,40,None]
        self.categories = ['under', 'between', 'between', 'between', 'between', 'over']
        super(ValueTransformer, self).__init__()

    def imputer(self, df, category, year,  age1, age2=None):
        if(year == 2021):
            true_operator = operator.mul
            # selecting the portion of the dataframe that has missing values in 2021 but not in 2019 for same records
            true_df = df.filter((F.col(self.col1).isNotNull()) & (F.col(self.col2).isNull()))
            col_name1 = self.col1
            col_name2 = self.col2
        elif(year == 2019):
            true_operator = operator.truediv
            # selecting a portion of the dataframe where missing values are present in 2019 but not in 2021
            true_df = df.filter((F.col(self.col1).isNull()) & (F.col(self.col2).isNotNull()))
            col_name1 = self.col2
            col_name2 = self.col1

        if(category == 'under'):
            # imputing the missing values in the required column for different age groups based on the value growth seen for respective groups
            true_df = true_df.withColumn(col_name2, F.when(F.col('Age') <= age1, \
                                                                                    F.round(true_operator(
                                                                                        F.col(col_name1), (1 +
                                                                                                                  self.variation_list[
                                                                                                                      0])),
                                                                                        3)). \
                                                            otherwise(F.col(col_name2)))
            true_df = true_df.filter(F.col('Age') <= age1)

        elif(category == 'between'):
            if(age2 == 25):
                index = 1
            elif(age2 == 30):
                index = 2
            elif(age2 == 35):
                index = 3
            elif(age2 == 40):
                index = 4
            true_df  = true_df.withColumn(col_name2, \
                                                               F.when((F.col('Age') > age1) & (F.col('Age') <= age2),
                                                                      F.round(true_operator(F.col(col_name1), (
                                                                                  1 + self.variation_list[index])), 3)). \
                                                               otherwise(F.col(col_name2)))
            true_df = true_df.filter((F.col('Age') > age1) & (F.col('Age') <= age2))

        elif(category == 'over'):
            true_df  = true_df.withColumn(col_name2, \
                                                               F.when(F.col('Age') > age1, \
                                                                      F.round(true_operator(F.col(col_name1), (
                                                                                  1 + self.variation_list[5])),3)). \
                                                               otherwise(F.col(col_name2)))
            true_df = true_df.filter(F.col('Age') > age1)
        return true_df

    def impute_separator(self, df, year):
        imputed_df = []
        for i, j, k in zip(self.age_1, self.age_2, self.categories):
          df_2019_not_null = self.imputer(df, category=k, year=year, age1=i, age2=j)
          imputed_df = imputed_df + [df_2019_not_null]
        df_not_null = self.dataframe_unifier(imputed_df)
        return df_not_null

    def dataframe_unifier(self, complete_dataframe_list):
        # taking out the last item in the list for merging later
        temp_df = complete_dataframe_list.pop(len(complete_dataframe_list) - 1)
        # merging all the dataframes in a given list
        for i in range(len(complete_dataframe_list)):
            if (len(complete_dataframe_list) > 0):
                temp_df = temp_df.unionAll(complete_dataframe_list.pop(len(complete_dataframe_list) - 1))
        return temp_df

    def _transform(self, df: DataFrame) -> DataFrame:
        #self.variation_calculator(df)
        # selecting a portion of the dataframe where players values are missing in both 2019 and 2021
        df_both_null = df.filter(((F.col(self.col1).isNull()) & (F.col(self.col2).isNull())))
        # selecting a portion of the dataframe where no missing values are present in both 2019 and 2021 for players
        df_both_not_null = df.filter((F.col(self.col1).isNotNull()) & (F.col(self.col2).isNotNull()))
        if self.num == 1:
          df_2019_not_null = self.impute_separator(df, year=2021)
          # we are not including records containing both null values, so we do not include df_both_null in the final dataframe
          final_df = [df_2019_not_null] + [df_both_not_null]
          final_df = self.dataframe_unifier(final_df)

        elif self.num == 2:
          df_2019_not_null = self.impute_separator(df, year=2021)
          df_2021_not_null = self.impute_separator(df, year=2019)
          final_df = [df_2019_not_null] + [df_2021_not_null] + [df_both_not_null]
          final_df = self.dataframe_unifier(final_df)

        regression_df = final_df
        return regression_df