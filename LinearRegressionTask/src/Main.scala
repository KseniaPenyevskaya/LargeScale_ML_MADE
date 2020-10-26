package hw_lr

import breeze.linalg._
import java.io.File

object Main {
  def main(args: Array[String]) : Unit = {
    val dataTrain = csvread(new File("train_data.csv"))
    val XTest = csvread(new File("test_data.csv"))
    val XTrain = dataTrain(::, 0 to -2)
    val yTrain = dataTrain(::, -1)

    val model = new LinearRegression()
    model.fit(XTrain, yTrain)
    var yPredict = DenseVector[Double](XTest.rows)
    yPredict = model.predict(XTest)
    csvwrite(new File("prediction.csv"), yPredict.toDenseMatrix.t)
  }
}
