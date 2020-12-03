package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame}
import breeze.linalg.{DenseMatrix, DenseVector, *}
import breeze.numerics._
import breeze.stats.mean


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val maxIter = 1000
  val learningRate = 0.01
  val delta = 0.1
  val eps = 0.1

  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val X: DenseMatrix[Double] = LinearRegressionTest._X
  lazy val y: DenseVector[Double] = LinearRegressionTest._y
  lazy val weights: linalg.DenseVector = LinearRegressionTest._weights
  lazy val bias: Double = LinearRegressionTest._bias

  private def validateModel(model: LinearRegressionModel,
                            weightsTrue : linalg.DenseVector,
                            biasTrue: Double,
                            delta: Double) = {
    val n = weightsTrue.size
    for(i <- 0 until n){
      model.weights(i) should be(weightsTrue(i) +- delta)
    }
    model.bias should be(biasTrue +- delta)
  }

  "Estimator" should "fit" in {
    val estimator: LinearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setMaxIter(maxIter)
      .setPredictionCol("prediction")

    val model = estimator.fit(data)
    validateModel(model, weights, bias, delta)
  }

  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights,
      bias
    ).setFeaturesCol("features")
      .setLabelCol("target")
      .setPredictionCol("prediction")

    val predictions = DenseVector(model.transform(data).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))
    val residual : DenseVector[Double] = predictions - y
    sqrt(mean(residual * residual)) should be(0.0 +- delta)
  }


  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("target")
        .setPredictionCol("prediction")
        .setMaxIter(maxIter)
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data).stages(0).asInstanceOf[LinearRegressionModel]
    validateModel(model, weights, bias, delta)


  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("target")
        .setPredictionCol("prediction")
        .setMaxIter(maxIter)
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val rereadModel = PipelineModel.load(tmpFolder.getAbsolutePath).stages(0).asInstanceOf[LinearRegressionModel]
    validateModel(rereadModel, weights, bias, eps)

    val predictions = DenseVector(rereadModel.transform(data).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))
    val residual : DenseVector[Double] = predictions - y
    sqrt(mean(residual * residual)) should be(0.0 +- delta)
  }
}

object LinearRegressionTest extends WithSpark {
  System.setProperty("hadoop.home.dir", "C:/winutils/")
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100, 4)
  lazy val _weights: linalg.DenseVector = Vectors.dense(1.1, 2.2, 3.3, 4.4).toDense
  lazy val _bias: Double = 5.5
  lazy val _y: DenseVector[Double] = _X * _weights.asBreeze + _bias

  lazy val _data: DataFrame = {
    import sqlc.implicits._

    val tmp = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)
    val df = tmp(*, ::)
      .iterator
      .map(x => (x(0), x(1), x(2), x(3), x(4)))
      .toSeq
      .toDF("x1", "x2", "x3", "x4", "target")

    val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3", "x4"))
      .setOutputCol("features")
    assembler.transform(df).select("features", "target")
  }
}
