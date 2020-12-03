package org.apache.spark.ml.made

import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{ParamMap, DoubleParam}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

// https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/regression/LinearRegression.scala

trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasPredictionCol with HasMaxIter{
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(labelCol))) {
      SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)
    } else {
      SchemaUtils.appendColumn(schema, StructField(getLabelCol, DoubleType))
    }
    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getLabelCol, new VectorUDT())
    } else {
      SchemaUtils.appendColumn(schema, StructField(getPredictionCol, new VectorUDT()))
    }
    schema
  }
  val learningRate: DoubleParam = new DoubleParam(this, "learningRate", "learning rate")
  def getLearningRate: Double = $(learningRate)
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("linearRegression"))

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setLearningRate(lr: Double): this.type = set(learningRate, lr)

  setDefault(maxIter, 500)
  setDefault(learningRate, 0.1)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val assembler = new VectorAssembler()
      .setInputCols(Array("bias", $(featuresCol), $(labelCol)))
      .setOutputCol("featuresAs")
    val features = assembler
      .transform(dataset.withColumn("bias", lit(1)))
      .select("featuresAs").as[Vector]

    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights = breeze.linalg.DenseVector.rand[Double](numFeatures + 1)
    val lr: Double = getLearningRate

    def optimizeWeights() : Unit = {
      val summary = features.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val features = v.asBreeze(0 to numFeatures).toDenseVector
          val target = v.asBreeze(-1)
          val gradients = features * (breeze.linalg.sum(features * weights) - target)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(gradients))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      val dw = summary.mean.asBreeze
      weights = weights - lr * dw
    }

    for (_ <- 0 to $(maxIter)) {
      optimizeWeights
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(weights(1 to numFeatures)).toDense, weights(0))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression] {
  System.setProperty("hadoop.home.dir", "C:/winutils/")
}

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights : DenseVector,
                                           val bias : Double)
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable{

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(uid, weights, bias))


  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val params = Seq(weights -> bias)
      sqlContext.createDataFrame(params).write.parquet(path + "/params")
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bWeights = weights.asBreeze
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x: Vector) => {
        Vectors.fromBreeze(breeze.linalg.DenseVector(bWeights.dot(x.asBreeze) + bias))
      })

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}
object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  System.setProperty("hadoop.home.dir", "C:/winutils/")
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val params = sqlContext.read.parquet(path + "/params")
      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val weights = params.select(params("_1").as[Vector]).first()
      val bias = params.select(params("_2")).first().getDouble(0)

      val model = new LinearRegressionModel(weights.toDense, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}
/*

trait StandardScalerParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class StandardScaler(override val uid: String) extends Estimator[StandardScalerModel] with StandardScalerParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("standardScaler"))

  override def fit(dataset: Dataset[_]): StandardScalerModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])

//    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
//      vectors.first().size
//    )

    val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
      val summarizer = new MultivariateOnlineSummarizer()
      data.foreach(v => summarizer.add(mllib.linalg.Vectors.fromBreeze(v.asBreeze)))
      Iterator(summarizer)
    }).reduce(_ merge _)

    copyValues(new StandardScalerModel(
      summary.mean.asML,
      Vectors.fromBreeze(breeze.numerics.sqrt(summary.variance.asBreeze)))).setParent(this)
//
//    val Row(row: Row) =  dataset
//      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
//      .first()
//
//    copyValues(new StandardScalerModel(row.getAs[Vector](0).toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[StandardScalerModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object StandardScaler extends DefaultParamsReadable[StandardScaler]

class StandardScalerModel private[made](
                           override val uid: String,
                           val means: DenseVector,
                           val stds: DenseVector) extends Model[StandardScalerModel] with StandardScalerParams with MLWritable {


  private[made] def this(means: Vector, stds: Vector) =
    this(Identifiable.randomUID("standardScalerModel"), means.toDense, stds.toDense)

  override def copy(extra: ParamMap): StandardScalerModel = copyValues(new StandardScalerModel(means, stds))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bMean = means.asBreeze
    val bStds = stds.asBreeze
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        Vectors.fromBreeze((x.asBreeze - bMean) /:/ bStds)
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = means.asInstanceOf[Vector] -> stds.asInstanceOf[Vector]

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object StandardScalerModel extends MLReadable[StandardScalerModel] {
  override def read: MLReader[StandardScalerModel] = new MLReader[StandardScalerModel] {
    override def load(path: String): StandardScalerModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val (means, std) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new StandardScalerModel(means, std)
      metadata.getAndSetParams(model)
      model
    }
  }
}

 */
