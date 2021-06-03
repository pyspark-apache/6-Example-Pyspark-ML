# 6- Example Pyspark ML

`[Age, Experience]` ---> new feature ---> independet feature

~~~python
from pyspark.ml.feature import VectorAssembler
featureassembler = VectorAssembler(inputCols=["age","Experience"],outputCol="Independent Features")
output=featureassembler.transform(training)
+---------+---+----------+------+--------------------+
|     Name|age|Experience|Salary|Independent Features|
+---------+---+----------+------+--------------------+
|    Krish| 31|        10| 30000|         [31.0,10.0]|
|Sudhanshu| 30|         8| 25000|          [30.0,8.0]|
|    Sunny| 29|         4| 20000|          [29.0,4.0]|
|     Paul| 24|         3| 20000|          [24.0,3.0]|
|   Harsha| 21|         1| 15000|          [21.0,1.0]|
|  Shubham| 23|         2| 18000|          [23.0,2.0]|
+---------+---+----------+------+--------------------+
~~~
~~~python
finalized_data=output.select("Independent Features","Salary")
+--------------------+------+
|Independent Features|Salary|
+--------------------+------+
|         [31.0,10.0]| 30000|
|          [30.0,8.0]| 25000|
|          [29.0,4.0]| 20000|
|          [24.0,3.0]| 20000|
|          [21.0,1.0]| 15000|
|          [23.0,2.0]| 18000|
+--------------------+------+
~~~

~~~python
from pyspark.ml.regression import LinearRegression
~~~
## train test split
~~~python
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Salary')
regressor=regressor.fit(train_data)
~~~
### Coefficients
~~~python

regressor.coefficients
DenseVector([-90.5483, 1608.7819])
~~~
### Intercepts
~~~python
regressor.intercept
16079.136690647425
~~~
~~~python
### Prediction
pred_results = regressor.evaluate(test_data)
+--------------------+------+-----------------+
|Independent Features|Salary|       prediction|
+--------------------+------+-----------------+
|          [23.0,2.0]| 18000|17214.09079632846|
+--------------------+------+-----------------+
pred_results.meanAbsoluteError, pred_results.meanSquaredError
(785.909203671541, 617653.2764156357)
~~~