package hw_lr

import breeze.linalg._

class LinearRegression {
  var weights = DenseVector[Double]()
  def fit(X : DenseMatrix[Double], y:DenseVector[Double]) : Unit = {
    weights = inv(X.t * X) * X.t * y
  }
  def predict(X : DenseMatrix[Double]) : DenseVector[Double] = {
    X * weights
  }
}
