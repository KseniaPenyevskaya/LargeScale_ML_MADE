package hw_tfidf

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import scala.collection.mutable.HashMap

object TFIDFtop100 {
  System.setProperty("hadoop.home.dir", "C:/winutils/")
  def main(args: Array[String]): Unit = {
    val fileInPath : String = "tripadvisor_hotel_reviews.csv"
    val fileOutPath : String = "output"
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("made-demo")
      .getOrCreate()
    import spark.implicits._

    // прочитаем датасет https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(fileInPath)
      .select(regexp_replace(lower(col("Review")), "[^\\w\\s-]", "").as("handled_text"))

    val nDocuments : Long = df.count

    def hashM(text:String) : HashMap[String, Int] = {
      var hMap = new HashMap[String, Int]()
      for (s <- text.split(" ")) {
        if (hMap.get(s) != None){
          hMap(s) += 1
        }
        else{
          hMap += (s -> 1)
        }
      }
      hMap
    }

    val dfHMap = df
      .map(x => hashM(x(0).toString()))
      .withColumn("documentID", monotonically_increasing_id())
      .select(col("documentID"), explode(col("value")).as(Array("word","number")))

    val top100IDF = dfHMap
      .groupBy(col("word"))
      .agg(count(col("documentID")) as "count")
      .orderBy(desc("count"))
      .limit(100)
      .select(col("word"), log(lit(nDocuments) / (col("count") + 1) + 1) as "idf")

    broadcast(top100IDF) //т.к. датасет по заданию маленький, его можно отправить воркерам
    val top100 = top100IDF.select(col("word")).collect.map(_.getString(0))

    val documentIDWindow = Window.partitionBy("documentID")
    val top100TF = dfHMap
      .withColumn("len", sum("number") over documentIDWindow)
      .filter(col("word") isin (top100:_*))
      .withColumn("tf", col("number") / col("len"))
      .select(col("word"), col("documentID"), col("tf"))

    val top100TFIDF = top100TF
      .join(top100IDF, "word")
      .withColumn("tfidf", col("tf") * col("idf"))
      .select(col("word"), col("documentID"), col("tfidf"))
    val pivotDF = top100TFIDF
      .groupBy("documentID")
      .pivot(col("word"))
      .sum("tfidf")
    //pivotDF.show(20)

    pivotDF
      .repartition(3)
      .write
      .mode("overwrite")
      .format("parquet")
      .option("header","true")
      .save(fileOutPath)
  }
}
