
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.joda.time

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.tree.impurity
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.mllib.tree.model.DecisionTreeModel

class ThreadExample() extends Thread{

}
object RunSVMWithSGD {

  def main(args: Array[String]): Unit = {

    SetLogger()

    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[*]"))

    //val (data1, data2, data3, data4, data5, testData, nbData, nbtestdata, kk1, kk2) = PrepareData(sc)
    val (nbData,nbtestdata,scalerddkk,scalerdd,labelpointrddkk,labelpointrdd)=PrepareData(sc)
    nbData.persist()
    nbtestdata.persist()
    scalerddkk.persist()
    scalerdd.persist()
    labelpointrddkk.persist()
    labelpointrdd.persist()


    do {
      println("是否将数据进行平衡处理(Y:是  N:否)？-------说明：是=》将数据正负样本平衡为1：1,否=》数据为原始状态，即为正负样本比例1：6")
      if (readLine() == "Y") {
        println("是否对数据进行标准化(Y:是  N:否)？-------说明：是=》对数据采用StandardScaler算法进行标准化，有统一标准差，否=》不进行标准化处理")
        if (readLine() == "Y") {
          println("是否进行五折交叉验证(Y:是  N:否)？--------说明：是=》对数据集划分五份，其中四份做训练集，一份做测试集，轮流进行，否=》将数据随机分为训练集：测试集0.8：0.2")
          if (readLine() == "Y") {
            println("111")
            //111
            //划分主体为ScaleRDDkk
            println("划分主体为ScaleRDDkk")
            val Array(data1,data2,data3,data4,data5)=scalerddkk.randomSplit(Array(0.2,0.2,0.2,0.2,0.2),123)
            val data1234 = data1 ++ data2 ++ data3 ++ data4
            val data1235 = data1 ++ data2 ++ data3 ++ data5
            val data1245 = data1 ++ data2 ++ data4 ++ data5
            val data1345 = data1 ++ data3 ++ data4 ++ data5
            val data2345 = data2 ++ data3 ++ data4 ++ data5


            val svmmodel11101 = SVMparametersTunning(data1234, data5)
            val svmmodel11102 = SVMparametersTunning(data1235, data4)
            val svmmodel11103 = SVMparametersTunning(data1245, data3)
            val svmmodel11104 = SVMparametersTunning(data1345, data2)
            val svmmodel11105 = SVMparametersTunning(data2345, data1)

            val lrmodel11101 = LRparametersTunning(data1234, data5)
            val lrmodel11102 = LRparametersTunning(data1235, data4)
            val lrmodel11103 = LRparametersTunning(data1245, data3)
            val lrmodel11104 = LRparametersTunning(data1345, data2)
            val lrmodel11105 = LRparametersTunning(data2345, data1)

            val treemodegini11101 = TreeGiniparametersTunning(data1234, data5)
            val treemodegini11102 = TreeGiniparametersTunning(data1235, data4)
            val treemodegini11103 = TreeGiniparametersTunning(data1245, data3)
            val treemodegini11104 = TreeGiniparametersTunning(data1345, data2)
            val treemodegini11105 = TreeGiniparametersTunning(data2345, data1)

            val treemodelentropy11101=TreeEntropyparametersTunning(data1234,data5)
            val treemodelentropy11102=TreeEntropyparametersTunning(data1235,data4)
            val treemodelentropy11103=TreeEntropyparametersTunning(data1245,data3)
            val treemodelentropy11104=TreeEntropyparametersTunning(data1345,data2)
            val treemodelentropy11105=TreeEntropyparametersTunning(data2345,data1)

            println("是否循环(Y:是  N：否)？")
          } else {
            println("110")
            //110
            //划分主体为ScaleRDDkk
            println("划分主体为ScaleRDDkk")
            val Array(trainData,validationsData)=scalerddkk.randomSplit(Array(0.8,0.2),111)

            val svmmodel11001=SVMparametersTunning(trainData,validationsData)

            val lrmodel11101=LRparametersTunning(trainData,validationsData)

            val treemodelgini11001=TreeGiniparametersTunning(trainData,validationsData)

            val treemodelentropy11001=TreeEntropyparametersTunning(trainData,validationsData)
            println("是否循环(Y:是  N：否)？")
          }
        } else {
          println("是否进行五折交叉验证(Y:是  N:否)？--------说明：是=》对数据集划分五份，其中四份做训练集，一份做测试集，轮流进行，否=》将数据随机分为训练集：测试集0.8：0.2")
          if (readLine() == "Y") {
            println("101") //101
            //将划分主体改为labelpointRDDkk
            println("划分主体为labelpointRDDkk")
            val Array(data1,data2,data3,data4,data5)=labelpointrddkk.randomSplit(Array(0.2,0.2,0.2,0.2,0.2),123)
            val data1234 = data1 ++ data2 ++ data3 ++ data4
            val data1235 = data1 ++ data2 ++ data3 ++ data5
            val data1245 = data1 ++ data2 ++ data4 ++ data5
            val data1345 = data1 ++ data3 ++ data4 ++ data5
            val data2345 = data2 ++ data3 ++ data4 ++ data5

            val svmmodel10101 = SVMparametersTunning(data1234, data5)
            val svmmodel10102 = SVMparametersTunning(data1235, data4)
            val svmmodel10103 = SVMparametersTunning(data1245, data3)
            val svmmodel10104 = SVMparametersTunning(data1345, data2)
            val svmmodel10105 = SVMparametersTunning(data2345, data1)

            val lrmodel10101 = LRparametersTunning(data1234, data5)
            val lrmodel10102 = LRparametersTunning(data1235, data4)
            val lrmodel10103 = LRparametersTunning(data1245, data3)
            val lrmodel10104 = LRparametersTunning(data1345, data2)
            val lrmodel10105 = LRparametersTunning(data2345, data1)

            val treemodegini10101 = TreeGiniparametersTunning(data1234, data5)
            val treemodegini10102 = TreeGiniparametersTunning(data1235, data4)
            val treemodegini10103 = TreeGiniparametersTunning(data1245, data3)
            val treemodegini10104 = TreeGiniparametersTunning(data1345, data2)
            val treemodegini10105 = TreeGiniparametersTunning(data2345, data1)

            val treemodelentropy10101=TreeEntropyparametersTunning(data1234,data5)
            val treemodelentropy10102=TreeEntropyparametersTunning(data1235,data4)
            val treemodelentropy10103=TreeEntropyparametersTunning(data1245,data3)
            val treemodelentropy10104=TreeEntropyparametersTunning(data1345,data2)
            val treemodelentropy10105=TreeEntropyparametersTunning(data2345,data1)
            println("是否循环(Y:是  N：否)？")
          } else {
            println("100")
            //100
            //划分主体为labelpointRDDkk
            println("划分主体为lebpointRDDkk")

            val Array(trainData,validationsData)=labelpointrddkk.randomSplit(Array(0.8,0.2),111)
            val svmmodel10001=SVMparametersTunning(trainData,validationsData)

            val lrmodel10101=LRparametersTunning(trainData,validationsData)

            val treemodelgini10001=TreeGiniparametersTunning(trainData,validationsData)

            val treemodelentropy10001=TreeEntropyparametersTunning(trainData,validationsData)
            println("是否循环(Y:是  N：否)？")
          }
        }

      } else {
        println("是否对数据进行标准化(Y:是  N:否)？-------说明：是=》对数据采用StandardScaler算法进行标准化，有统一标准差，否=》不进行标准化处理")
        if (readLine() == "Y") {
          println("是否进行五折交叉验证(Y:是  N:否)？--------说明：是=》对数据集划分五份，其中四份做训练集，一份做测试集，轮流进行，否=》将数据随机分为训练集：测试集0.8：0.2")
          if (readLine() == "Y") {
            println("011")
            //011
            //划分主体为ScaleRDD
            println("划分主体为ScaleRDD")
            val Array(data1,data2,data3,data4,data5)=scalerdd.randomSplit(Array(0.2,0.2,0.2,0.2,0.2),123)
            val data1234 = data1 ++ data2 ++ data3 ++ data4
            val data1235 = data1 ++ data2 ++ data3 ++ data5
            val data1245 = data1 ++ data2 ++ data4 ++ data5
            val data1345 = data1 ++ data3 ++ data4 ++ data5
            val data2345 = data2 ++ data3 ++ data4 ++ data5

            val svmmodel01101 = SVMparametersTunning(data1234, data5)
            val svmmodel01102 = SVMparametersTunning(data1235, data4)
            val svmmodel01103 = SVMparametersTunning(data1245, data3)
            val svmmodel01104 = SVMparametersTunning(data1345, data2)
            val svmmodel01105 = SVMparametersTunning(data2345, data1)

            val lrmodel01101 = LRparametersTunning(data1234, data5)
            val lrmodel01102 = LRparametersTunning(data1235, data4)
            val lrmodel01103 = LRparametersTunning(data1245, data3)
            val lrmodel01104 = LRparametersTunning(data1345, data2)
            val lrmodel01105 = LRparametersTunning(data2345, data1)

            val treemodegini01101 = TreeGiniparametersTunning(data1234, data5)
            val treemodegini01102 = TreeGiniparametersTunning(data1235, data4)
            val treemodegini01103 = TreeGiniparametersTunning(data1245, data3)
            val treemodegini01104 = TreeGiniparametersTunning(data1345, data2)
            val treemodegini01105 = TreeGiniparametersTunning(data2345, data1)

            val treemodelentropy01101=TreeEntropyparametersTunning(data1234,data5)
            val treemodelentropy01102=TreeEntropyparametersTunning(data1235,data4)
            val treemodelentropy01103=TreeEntropyparametersTunning(data1245,data3)
            val treemodelentropy01104=TreeEntropyparametersTunning(data1345,data2)
            val treemodelentropy01105=TreeEntropyparametersTunning(data2345,data1)
            println("是否循环(Y:是  N：否)？")
          } else {
            println("010")
            //010
            //划分主体为ScaleRDD
            println("划分主体为ScaleRDD")
            val Array(trainData,validationsData)=scalerdd.randomSplit(Array(0.8,0.2),111)
            val svmmodel01001=SVMparametersTunning(trainData,validationsData)

            val lrmodel01001=LRparametersTunning(trainData,validationsData)

            val treemodelgini01001=TreeGiniparametersTunning(trainData,validationsData)

            val treemodelentropy01001=TreeEntropyparametersTunning(trainData,validationsData)
            println("是否循环(Y:是  N：否)？")
          }
        }
        println("是否进行五折交叉验证(Y:是  N:否)？--------说明：是=》对数据集划分五份，其中四份做训练集，一份做测试集，轮流进行，否=》将数据随机分为训练集：测试集0.8：0.2")
        if (readLine() == "Y") {
          println("001")
          //001
          //划分主体为labelpointRDD
          println("划分主体为labelpointRDD")
          val Array(data1,data2,data3,data4,data5)=labelpointrdd.randomSplit(Array(0.2,0.2,0.2,0.2,0.2),123)
          val data1234 = data1 ++ data2 ++ data3 ++ data4
          val data1235 = data1 ++ data2 ++ data3 ++ data5
          val data1245 = data1 ++ data2 ++ data4 ++ data5
          val data1345 = data1 ++ data3 ++ data4 ++ data5
          val data2345 = data2 ++ data3 ++ data4 ++ data5

          val svmmodel00101 = SVMparametersTunning(data1234, data5)
          val svmmodel00102 = SVMparametersTunning(data1235, data4)
          val svmmodel00103 = SVMparametersTunning(data1245, data3)
          val svmmodel00104 = SVMparametersTunning(data1345, data2)
          val svmmodel00105 = SVMparametersTunning(data2345, data1)

          val lrmodel00101 = LRparametersTunning(data1234, data5)
          val lrmodel00102 = LRparametersTunning(data1235, data4)
          val lrmodel00103 = LRparametersTunning(data1245, data3)
          val lrmodel00104 = LRparametersTunning(data1345, data2)
          val lrmodel00105 = LRparametersTunning(data2345, data1)


          val treemodegini00101 = TreeGiniparametersTunning(data1234, data5)
          val treemodegini00102 = TreeGiniparametersTunning(data1235, data4)
          val treemodegini00103 = TreeGiniparametersTunning(data1245, data3)
          val treemodegini00104 = TreeGiniparametersTunning(data1345, data2)
          val treemodegini00105 = TreeGiniparametersTunning(data2345, data1)

          val treemodelentropy00101=TreeEntropyparametersTunning(data1234,data5)
          val treemodelentropy00102=TreeEntropyparametersTunning(data1235,data4)
          val treemodelentropy00103=TreeEntropyparametersTunning(data1245,data3)
          val treemodelentropy00104=TreeEntropyparametersTunning(data1345,data2)
          val treemodelentropy00105=TreeEntropyparametersTunning(data2345,data1)
          println("是否循环(Y:是  N：否)？")
        } else {
          println("000")
          //000
          //划分主体为labelpointRDD
          println("划分主体为labelpointRDD")
          val Array(trainData,validationsData)=labelpointrdd.randomSplit(Array(0.8,0.2),111)
          val svmmodel00001=SVMparametersTunning(trainData,validationsData)

          val lrmodel00001=LRparametersTunning(trainData,validationsData)

          val treemodelgini00001=TreeGiniparametersTunning(trainData,validationsData)

          val treemodelentropy00001=TreeEntropyparametersTunning(trainData,validationsData)
          println("是否循环(Y:是  N：否)？")
        }
      }
    } while (readLine() == "Y")
    println("是否进行贝叶斯模型训练（Y：是 N：否）？")
    if (readLine()=="Y") {
      println("贝叶斯模型：")
      val nbmodel = NBparametersTunning(nbData, nbtestdata)
    }else{}

    //最优模型测试
    /*println("模型测试（举例采用model11101模型）：")
    val Array(testdata)=scalerddkk.randomSplit(Array(0.1))
    val Array(data1,data2,data3,data4,data5)=scalerddkk.randomSplit(Array(0.2,0.2,0.2,0.2,0.2),123)
    val data1234 = data1 ++ data2 ++ data3 ++ data4
    val data1235 = data1 ++ data2 ++ data3 ++ data5
    val data1245 = data1 ++ data2 ++ data4 ++ data5
    val data1345 = data1 ++ data3 ++ data4 ++ data5
    val data2345 = data2 ++ data3 ++ data4 ++ data5

    val svmmodel11101 = SVMparametersTunning(data1234, data5)
    TestModel(svmmodel11101,testdata)*/

    nbData.unpersist()
    nbtestdata.unpersist()
    scalerdd.unpersist()
    scalerddkk.unpersist()
    labelpointrdd.unpersist()
    labelpointrddkk.unpersist()



  }
  //数据准备函数
  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint],
    RDD[LabeledPoint],RDD[LabeledPoint],RDD[LabeledPoint],RDD[LabeledPoint]) = {

    val rawData = sc.textFile("/home/fjf/pssm8.txt")
    val lines = rawData.map(_.split("\t"))//压扁

    println("正负样本总共：" + lines.count.toString() + "条")

    val labelpointRDD = lines.map { fields =>
      val numericalFeatures = fields.slice(1, fields.size - 1)
        .map(d =>d.toDouble)//特征变量

    //val kk=numericalFeatures.map(d=>normalization(d))
    val label = fields(fields.size - 1).toInt//便签变量
      LabeledPoint(label, Vectors.dense(numericalFeatures))
    }
    val featuresData = labelpointRDD.map(labelpoint => labelpoint.features)
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresData)
    val scaledRDD = labelpointRDD.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)))
    //labelpointRDD.saveAsTextFile("/home/fjf/pssm99.txt")

    val featuresDataNormalizer = labelpointRDD.map(labelpoint => labelpoint.features)
    //val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresData)
    val normalizer=new Normalizer(4)
    val NormalizerRDD = labelpointRDD.map(labelpoint => LabeledPoint(labelpoint.label, normalizer.transform(labelpoint.features)))

    val rawData1 = sc.textFile("/home/fjf/pssm1.txt")
    val lines1 = rawData1.map(_.split("\t"))//压扁

    println("正样本总共：" + lines1.count.toString() + "条")

    val labelpointRDD1 = lines.map { fields =>
      val numericalFeatures = fields.slice(1, fields.size - 1)
        .map(d =>d.toDouble)//特征变量

    val label = fields(fields.size - 1).toInt//便签变量
      LabeledPoint(label, Vectors.dense(numericalFeatures))
    }
    val featuresData1 = labelpointRDD1.map(labelpoint => labelpoint.features)
    val stdScaler1 = new StandardScaler(withMean = true, withStd = true).fit(featuresData1)
    val scaledRDD1 = labelpointRDD1.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler1.transform(labelpoint.features)))


    val rawData0 = sc.textFile("/home/fjf/pssm0.txt")
    val lines0 = rawData0.map(_.split("\t"))//压扁
    println("负样本总共：" + lines0.count.toString() + "条")

    val labelpointRDD0 = lines0.map { fields =>
      val numericalFeatures = fields.slice(1, fields.size - 1)
        .map(d =>d.toDouble)//特征变量
    val label = fields(fields.size - 1).toInt//便签变量
      LabeledPoint(label, Vectors.dense(numericalFeatures))
    }
    val featuresData0 = labelpointRDD0.map(labelpoint => labelpoint.features)
    val stdScaler0 = new StandardScaler(withMean = true, withStd = true).fit(featuresData0)
    val scaledRDD0 = labelpointRDD0.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler0.transform(labelpoint.features)))


    val  NBData = lines.map { r =>
      val label =r(r.size - 1).toInt
      val features = r.slice(1, r.size - 1).map(d => d.toDouble).map(d => normalization(d))
      LabeledPoint(label, Vectors.dense(features))
    }

    val Array(nbdata,nbtestdata)=NBData.randomSplit(Array(0.9,0.1),123)

    val Array(dataN)=scaledRDD0.randomSplit(Array(0.2),123)
    val labelpointRDDkk=dataN++scaledRDD1

    val featureskk=labelpointRDDkk.map(labelpoint=>labelpoint.features)
    val stdScalerkk=new StandardScaler(withMean=true,withStd = true).fit(featureskk)
    val scaledRDDkk=labelpointRDDkk.map(labelpoint=>LabeledPoint(labelpoint.label,stdScalerkk.transform(labelpoint.features)))


    (nbdata,nbtestdata,scaledRDDkk,scaledRDD,labelpointRDDkk,labelpointRDD)
  }
  def normalization(x: Double):(Double)={
    val min= -12
    val max= 13
    val normaliz=(x-min)/(max-min)
    (normaliz)
  }
/*
  def SlidingWindows(rowpssm:RDD[String],slidingnum:Int):(RDD[LabeledPoint])={
    val lines=rowpssm.flatMap(_.split("\t"))

  }
*/

  def trainLRmodel(trainData:RDD[LabeledPoint],numIterations:Int):(LogisticRegressionModel)={
    val LRmodel=LogisticRegressionWithSGD.train(trainData,numIterations)
    (LRmodel)
  }


  def trainTreemodel(trainData:RDD[LabeledPoint],impurity: Impurity,maxTreeDepth:Int):(DecisionTreeModel)={
    val Treemodel=DecisionTree.train(trainData,Algo.Classification,impurity,maxTreeDepth)
    (Treemodel)
  }
  def trainSVMModel(trainData: RDD[LabeledPoint], numIterations: Int, stepSize: Double, regParam: Double): (SVMModel) = {
    val SVMmodel = SVMWithSGD.train(trainData, numIterations, stepSize, regParam)
    (SVMmodel)
  }
 def trainNBModel(trainData: RDD[LabeledPoint],lambda:Double): (ClassificationModel) = {
    val NBmodel=NaiveBayes.train(trainData,lambda)
    (NBmodel)
  }

  def Accuracy(model:ClassificationModel,validationData:RDD[LabeledPoint]):(Double)={
    val TotalCorrect = validationData.map { point =>
      if (model.predict(point.features) == point.label) 1 else 0
    }.sum

    val Accuracy = TotalCorrect / validationData.count
    //println("逻:辑回归平均分类正确率: " + Accuracy)
    (Accuracy)
  }

  def TreeAccuracy(model:DecisionTreeModel,validationData:RDD[LabeledPoint]):(Double)={
    val TotalCorrect = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum

    val TreeAccuracy = TotalCorrect / validationData.count
    //println("逻:辑回归平均分类正确率: " + Accuracy)
    (TreeAccuracy)
  }
  def MCC(model:ClassificationModel,validationData:RDD[LabeledPoint]):(Double)={
    val TP = validationData.map { point =>
      if (model.predict(point.features) == 1 &&  point.label==1) 1 else 0
    }.sum
    val FN = validationData.map { point =>
      if (model.predict(point.features) == 1 &&point.label==0) 1 else 0
    }.sum
    val TN = validationData.map { point =>
      if (model.predict(point.features) == 0 &&point.label==0) 1 else 0
    }.sum
    val FP = validationData.map { point =>
      if (model.predict(point.features) == 0 && point.label==1) 1 else 0
    }.sum
    val aa=TP*TN-FP*FN
    val bb=(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    val cc=math.sqrt(bb)
    val MCC=aa/cc
    (MCC)
  }
  def P(model:ClassificationModel,validationData:RDD[LabeledPoint]):(Double)={
    val TP = validationData.map { point =>
      if (model.predict(point.features) == 1 &&  point.label==1) 1 else 0
    }.sum
    val FN = validationData.map { point =>
      if (model.predict(point.features) == 1 &&point.label==0) 1 else 0
    }.sum
    val TN = validationData.map { point =>
      if (model.predict(point.features) == 0 &&point.label==0) 1 else 0
    }.sum
    val FP = validationData.map { point =>
      if (model.predict(point.features) == 0 && point.label==1) 1 else 0
    }.sum
    val aa=TP+FP
    val P=TP/aa
    (P)
  }
  def R(model:ClassificationModel,validationData:RDD[LabeledPoint]):(Double)={
    val TP = validationData.map { point =>
      if (model.predict(point.features) == 1 &&  point.label==1) 1 else 0
    }.sum
    val FN = validationData.map { point =>
      if (model.predict(point.features) == 1 &&point.label==0) 1 else 0
    }.sum
    val TN = validationData.map { point =>
      if (model.predict(point.features) == 0 &&point.label==0) 1 else 0
    }.sum
    val FP = validationData.map { point =>
      if (model.predict(point.features) == 0 && point.label==1) 1 else 0
    }.sum
    val aa=TP+FN
    val R=TP/aa
    (R)
  }
  def TreeP(model:DecisionTreeModel,validationData:RDD[LabeledPoint]):(Double)={
    val TP = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 1 &&  point.label==1) 1 else 0
    }.sum
    val FN = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 1 &&  point.label==0) 1 else 0
    }.sum
    val TN = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 0 &&  point.label==0) 1 else 0
    }.sum
    val FP = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 0 &&  point.label==1) 1 else 0
    }.sum
    val aa=TP+FP
    val TreeP=TP/aa
    (TreeP)
  }
  def TreeR(model:DecisionTreeModel,validationData:RDD[LabeledPoint]):(Double)={
    val TP = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 1 &&  point.label==1) 1 else 0
    }.sum
    val FN = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 1 &&  point.label==0) 1 else 0
    }.sum
    val TN = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 0 &&  point.label==0) 1 else 0
    }.sum
    val FP = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 0 &&  point.label==1) 1 else 0
    }.sum
    val aa=TP+FN
    val TreeR=TP/aa
    (TreeR)
  }
  def TreeMCC(model:DecisionTreeModel,validationData:RDD[LabeledPoint]):(Double)={
    val TP = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 1 &&  point.label==1) 1 else 0
    }.sum
    val FN = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 1 &&  point.label==0) 1 else 0
    }.sum
    val TN = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 0 &&  point.label==0) 1 else 0
    }.sum
    val FP = validationData.map { point =>
      val score = model.predict(point.features)
      // 决策树的预测阈值需要明确给出
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == 0 &&  point.label==1) 1 else 0
    }.sum
    val aa=TP*TN-FP*FN
    val bb=(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    val cc=math.sqrt(bb)
    val TreeMCC=aa/cc
    (TreeMCC)
  }

  def PR(model:ClassificationModel,validationData:RDD[LabeledPoint]):(Double)={
    val scoreAndLabels = validationData.map { data =>
      var predict1 = model.predict(data.features)
      (predict1, data.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    //val AUC = Metrics.areaUnderROC
    val pr=Metrics.precisionByThreshold()
    pr.collect().foreach{println}
    val PR=Metrics.areaUnderPR()
    (PR)
  }
  def TreePR(model:DecisionTreeModel,validationData:RDD[LabeledPoint]):(Double)={
    val scoreAndLabels = validationData.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    //val AUC = Metrics.areaUnderROC

    val TreePR=Metrics.areaUnderPR()
    (TreePR)
  }
  def AUC(model: ClassificationModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { data =>
      var predict1 = model.predict(data.features)
      (predict1, data.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val AUC = Metrics.areaUnderROC
    (AUC)
  }
  def TreeAUC(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val TreeAUC = Metrics.areaUnderROC
    (TreeAUC)
  }

  def TestModel(model: ClassificationModel, testData: RDD[LabeledPoint]): Unit = {
    println("其中前10条的预测结果如下：")
    val testPredictData = testData.take(10)
    testPredictData.foreach { test =>
      val predict = model.predict(test.features)
      val result = (if (test.label == predict) "正确" else "错误")
      //val count=0
      // if (test.label==predict)count+1
      println("实际結果:" + test.label + "預測結果:" + predict + result +",特征"+ test.features)
    }
  }
  def TestTreeModel(model: DecisionTreeModel, testData: RDD[LabeledPoint]): Unit = {
    println("其中前10条的预测结果如下：")
    val testPredictData = testData.take(10)
    testPredictData.foreach { test =>
      val predict = model.predict(test.features)
      val result = (if (test.label == predict) "正确" else "错误")
      //val count=0
      // if (test.label==predict)count+1
      println("实际结果:" + test.label + "预测结果:" + predict + result +",特征"+ test.features)
    }
  }

  def SVMparametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): ClassificationModel = {
    println("-----SVM交叉评估所有参数---------参数采用：numIterations(1, 3, 5, 15, 25),stepSize(10, 50, 100, 200), regParam(0.01, 0.1, 1)")
    val bestModel = SVMevaluateAllParameter(trainData, validationData, Array(1, 3, 5, 15, 25),
      Array(10, 50, 100, 200), Array(0.01, 0.1, 1))
    (bestModel)
  }
  def SVMevaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], numIterationsArray: Array[Int],
                           stepSizeArray: Array[Double], regParamArray: Array[Double]): SVMModel =
  {
    val evaluations =
      for (numIterations <- numIterationsArray; stepSize <- stepSizeArray; regParam <- regParamArray) yield {
        val (model) = trainSVMModel(trainData, numIterations, stepSize, regParam)
        val auc = AUC(model, validationData)
        val SVMacc=Accuracy(model,validationData)
        val SVMmcc=MCC(model,validationData)
        val SVMP=P(model,validationData)
        val SVMR=R(model,validationData)
        (numIterations, stepSize, regParam, auc,SVMacc,SVMmcc,SVMP,SVMR)
      }
    //println(evaluations)
    val dd=evaluations.toSeq
    dd.foreach{ case (numIterations, stepSize, regParam,auc,acc,mcc,p,r) =>
      println(f"参数：迭代次数$numIterations,参数：步长  $stepSize,参数：正则化  $regParam,评价：AUC  $auc,评价：ACC  $acc,评价：MCC   $mcc,评价： P$p,评价：R   $r ")
    }
    val BestEval = (evaluations.sortBy(_._4).reverse)(0)

    println("最佳参数SVM：numIterations:" + BestEval._1 + "  ,stepSize:" + BestEval._2 + "  ,regParam:" + BestEval._3
      + "  ,使用validationData评估,AUC = " + BestEval._4,"ACC:  "+BestEval._5+",MCC:  "+BestEval._6+",P:   "+BestEval._7+",R:   "+BestEval._8)

    val bestModel = SVMWithSGD.train(
      trainData.union(validationData), BestEval._1, BestEval._2, BestEval._3)
    //val SVMacc=Accuracy(bestModel,validationData)
    (bestModel)
  }
  def LRparametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): ClassificationModel = {
    println("-----LR交叉评估所有参数---------参数采用：numIterations(1, 3, 5, 15, 25,50,100)")
    val LRbestModel = LRevaluateAllParameter(trainData, validationData, Array(1, 3, 5, 15, 25,50,100))
    (LRbestModel)
  }
  def LRevaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], numIterationsArray: Array[Int]
                              ): LogisticRegressionModel =
  {
    val evaluations =
      for (numIterations <- numIterationsArray) yield {
        val (model) = trainLRmodel(trainData, numIterations)
        val auc = AUC(model, validationData)
        val SVMacc=Accuracy(model,validationData)
        val SVMmcc=MCC(model,validationData)
        val SVMP=P(model,validationData)
        val SVMR=R(model,validationData)
        (numIterations, auc,SVMacc,SVMmcc,SVMP,SVMR)
      }
    //println(evaluations)
    val dd=evaluations.toSeq
    dd.foreach{ case (numIterations,auc,acc,mcc,p,r) =>
      println(f"参数：迭代次数$numIterations,评价：AUC  $auc,评价：ACC  $acc,评价：MCC   $mcc,评价： P$p,评价：R   $r ")
    }
    val BestEval = (evaluations.sortBy(_._2).reverse)(0)

    println("最佳参数LR：numIterations:" + BestEval._1 + "  ,使用validationData评估,AUC = " + BestEval._2,"ACC:  "+BestEval._3+",MCC:  "+BestEval._4+",P:   "+BestEval._5+",R:   "+BestEval._6)

    val LRbestModel = LogisticRegressionWithSGD.train(
      trainData.union(validationData), BestEval._1)

    (LRbestModel)
  }


  def NBparametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): ClassificationModel = {
    println("-----NB交叉评估所有参数---------参数采用：lambda(0.001, 0.01, 0.1, 1, 10)")
    val NBbestModel = NBevaluateAllParameter(trainData, validationData, Array(0.001, 0.01, 0.1, 1, 10))
    (NBbestModel)
  }
  def NBevaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], lambdaArray: Array[Double]
                            ): ClassificationModel =
  {
    val evaluations =
      for (lambda <- lambdaArray) yield {
        val (model) = trainNBModel(trainData, lambda)
        val auc = AUC(model, validationData)
        val NBacc=Accuracy(model,validationData)
        val NBmcc=MCC(model,validationData)
        val NBP=P(model,validationData)
        val NBR=R(model,validationData)
        (lambda, auc,NBacc,NBmcc,NBP,NBR)
      }
    //println(evaluations)
    val dd=evaluations.toSeq
    dd.foreach{ case (numIterations,auc,acc,mcc,p,r) =>
      println(f"参数：lambda $numIterations,评价：AUC  $auc,评价：ACC  $acc,评价：MCC   $mcc,评价： P$p,评价：R   $r ")
    }
    val BestEval = (evaluations.sortBy(_._2).reverse)(0)

    println("最佳参数NB：lambda" + BestEval._1 + "  ,使用validationData评估,AUC = " + BestEval._2,"ACC:  "+BestEval._3+",MCC:  "+BestEval._4+",P:   "+BestEval._5+",R:   "+BestEval._6)

    val NBbestModel = NaiveBayes.train(
      trainData.union(validationData), BestEval._1)

    (NBbestModel)
  }
  def TreeEntropyparametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    println("-----TreeEntropy交叉评估所有参数---------参数采用：Impurity(Entropy),maxTreeDepth(1, 2, 3, 4, 5, 10, 20)")
    val TreebestModel = TreeEntropyevaluateAllParameter(trainData, validationData, Array(1, 2, 3, 4, 5, 10, 20))
    (TreebestModel)
  }
  def TreeEntropyevaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], maxTreeDepthArray: Array[Int]
                            ): DecisionTreeModel =
  {
    val evaluations =
      for (maxTreeDepth <- maxTreeDepthArray) yield {
        val (model) = trainTreemodel(trainData,Entropy,maxTreeDepth)
        val auc = TreeAUC(model, validationData)
        val Treeacc=TreeAccuracy(model,validationData)
        val Treemcc=TreeMCC(model,validationData)
        val Treep=TreeP(model,validationData)
        val Treer=TreeR(model,validationData)
        (maxTreeDepth, auc,Treeacc,Treemcc,Treep,Treer)
      }
    //println(evaluations)
    val dd=evaluations.toSeq
    dd.foreach{ case (numIterations,auc,acc,mcc,p,r) =>
      println(f"参数：maxTreeDepth $numIterations,评价：AUC  $auc,评价：ACC  $acc,评价：MCC   $mcc,评价： P$p,评价：R   $r ")
    }
    val BestEval = (evaluations.sortBy(_._2).reverse)(0)

    println("最佳参数TreeEntropy：maxTreeDepth  " + BestEval._1 + "  ,使用validationData评估,AUC = " + BestEval._2,"ACC:  "+BestEval._3+",MCC:  "+BestEval._4+",P:   "+BestEval._5+",R:   "+BestEval._6)

    val TreebestModel = DecisionTree.train(
      trainData.union(validationData), Algo.Classification,Entropy,BestEval._1)

    (TreebestModel)
  }
  def TreeGiniparametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    println("-----TreeGini交叉评估所有参数---------参数采用：Impurity(Gini),maxTreeDepth(1, 2, 3, 4, 5, 10, 20)")
    val TreebestModel = TreeGinievaluateAllParameter(trainData, validationData, Array(1, 2, 3, 4, 5, 10, 20))
    (TreebestModel)
  }
  def TreeGinievaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], maxTreeDepthArray: Array[Int]
                                  ): DecisionTreeModel =
  {
    val evaluations =
      for (maxTreeDepth <- maxTreeDepthArray) yield {
        val (model) = trainTreemodel(trainData,Gini,maxTreeDepth)
        val auc = TreeAUC(model, validationData)
        val Treeacc=TreeAccuracy(model,validationData)
        val Treemcc=TreeMCC(model,validationData)
        val Treep=TreeP(model,validationData)
        val Treer=TreeR(model,validationData)
        (maxTreeDepth, auc,Treeacc,Treemcc,Treep,Treer)
      }
    //println(evaluations)
    val dd=evaluations.toSeq
    dd.foreach{ case (numIterations,auc,acc,mcc,p,r) =>
      println(f"参数：maxTreeDepth $numIterations,评价：AUC  $auc,评价：ACC  $acc,评价：MCC   $mcc,评价： P$p,评价：R   $r ")
    }
    val BestEval = (evaluations.sortBy(_._2).reverse)(0)

    println("最佳参数TreeGini：maxTreeDepth  " + BestEval._1 + "  ,使用validationData评估,AUC = " + BestEval._2,"ACC:  "+BestEval._3+",MCC:  "+BestEval._4+",P:   "+BestEval._5+",R:   "+BestEval._6)

    val TreebestModel = DecisionTree.train(
      trainData.union(validationData), Algo.Classification,Gini,BestEval._1)

    (TreebestModel)
  }





  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}


