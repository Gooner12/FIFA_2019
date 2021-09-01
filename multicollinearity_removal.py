from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

class MultiCollinearityRemover():

  def __init__(self, vif_threshold=5):
    self.vif_threshold = vif_threshold
    self.max_col_index = 10000 # setting an arbitrary value
    super(MultiCollinearityRemover, self).__init__()

  def vif_setter(self, df):
    # global vif_max
    # global colnum_max
    col_names = df.columns
    # max_col_index = 10000 # setting an arbitrary value
    vif_max = self.vif_threshold + 1

    while vif_max > 5:
      vif_max, max_col_index = self.vif_calc(df, col_names, vif_max, self.max_col_index, self.vif_threshold)
      if vif_max > self.vif_threshold:
        print('The removed column is ',df[max_col_index])
        df = df.drop(df[max_col_index])
        col_names = df.columns
      else:
        return df


  def vif_calc(self, df, col_names, vif_max, max_col_index, vif_threshold):
    print('The dataframe has %d rows and %d columns'%(df.count(), len(col_names)))
    vif_max = vif_threshold
    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
    for i in range(0,len(col_names)):
      # creating a feature vector with one independent variable as a target
      temp_df = df.rdd.map(lambda x: [Vectors.dense(x[0:i]+x[i+1:]), x[i]]).toDF(['features', 'label'])
      temp_df.show(5)
      lr = LinearRegression(featuresCol = 'features', labelCol = 'label')
      lr_model = lr.fit(temp_df)
      predictions = lr_model.transform(temp_df)
      R2 = evaluator.evaluate(predictions, {evaluator.metricName: 'r2'})
      vif = 1/(1-R2)
      if vif_max < vif:
        vif_max = vif
        max_col_index = i
    return vif_max, max_col_index