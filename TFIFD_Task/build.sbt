name := "TrySparkScala"

version := "0.1"

scalaVersion := "2.11.7"

libraryDependencies  ++= Seq(
  "org.apache.spark"%% "spark-core" % "2.4.0",
  "org.apache.spark" %% "spark-sql" % "2.4.0"
)
/*
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.0"
 */