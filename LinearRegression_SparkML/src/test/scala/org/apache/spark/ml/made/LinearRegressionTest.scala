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
/*
class StandardScalerTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val data: DataFrame = StandardScalerTest._data
  lazy val vectors: Seq[Vector] = StandardScalerTest._vectors

  "Model" should "scale input data" in {
    val model: StandardScalerModel = new StandardScalerModel(
      means = Vectors.dense(2.0, -0.5).toDense,
      stds = Vectors.dense(1.5, 0.5).toDense
    ).setInputCol("features")
      .setOutputCol("features")


    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5 - 2.0) / 1.5 +- delta)
    vectors(0)(1) should be((12 + 0.5) / 0.5 +- delta)

    vectors(1)(0) should be((-1 - 2.0) / 1.5 +- delta)
    vectors(1)(1) should be((0 + 0.5) / 0.5 +- delta)
  }

  "Estimator" should "calculate means" in {
    val estimator = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    model.means(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    model.means(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
  }

  "Estimator" should "calculate stds" in {
    val estimator = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

//    model.stds(0) should be(Math.sqrt(vectors.map(v => v(0) * v(0)).sum / vectors.length - model.means(0) * model.means(0)) +- delta)
//    model.stds(1) should be(Math.sqrt(vectors.map(v => v(1) * v(1)).sum / vectors.length - model.means(1) * model.means(1)) +- delta)
    model.stds(0) should be(Math.sqrt(vectors.map(v => (v(0) -  model.means(0))).map(x => x*x).sum / (vectors.length - 1)) +- delta)
    model.stds(1) should be(Math.sqrt(vectors.map(v => (v(1) -  model.means(1))).map(x => x*x).sum / (vectors.length - 1)) +- delta)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  private def validateModel(model: StandardScalerModel, data: DataFrame) = {
    val vectors: Array[Vector] = data.collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5 - model.means(0)) / model.stds(0) +- delta)
    vectors(0)(1) should be((12 - model.means(1)) / model.stds(1) +- delta)

    vectors(1)(0) should be((-1 - model.means(0)) / model.stds(0) +- delta)
    vectors(1)(1) should be((0 - model.means(1)) / model.stds(1) +- delta)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new StandardScaler()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data).stages(0).asInstanceOf[StandardScalerModel]

    model.means(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    model.means(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new StandardScaler()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[StandardScalerModel], reRead.transform(data))
  }
}

object StandardScalerTest extends WithSpark {

  lazy val _vectors = Seq(
    Vectors.dense(13.5, 12),
    Vectors.dense(-1, 0)
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }
}

 */
