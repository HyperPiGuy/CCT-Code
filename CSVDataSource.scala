package toolkit.neuralnetwork.source

import libcog._
import toolkit.neuralnetwork.DifferentiableField

/**
* The CSVDataSource will read in data from a CSV file and populate a field
* with the data with each element being the Shape of elemSize
*
*@param fileName CSV file to load
*@param batchSize # of examples to read from file
*@param header True if CSV file has a header row on top of file, false if not
*@param areNums True if all data in file are numbers, False if data includes char data
*@param elemSize Shape of data element
*
* Created by Matthew Fontaine
*/

case class CSVDataSource(fileName:String,
                         override val batchSize:Int,
                         header:Boolean, 
                         areNums:Boolean,
                         elemSize:Shape) extends DifferentiableField {

    override val forward: Field = {

      ////////////////////////////////read in csv data to Array
      val bufferedSource = io.Source.fromFile(fileName)
      val dataIn: Array[Array[String]] = {
        var mat: Array[Array[String]] = Array.empty
        val takeElem: Int =  header match{
          case true => batchSize+1
          case false => batchSize
          }
        var skipHeader = header
        for (line <- bufferedSource.getLines.take(takeElem)) {
          if (skipHeader) {
            skipHeader = false
          }
          else {
            val cols = line.split(",").map(_.trim)
            mat = mat :+ cols
          }
        }
        mat
      }
      bufferedSource.close

      require(dataIn.length>=batchSize,s"${dataIn.length} examples found in file. This is not enough for the $batchSize batchSize")

      val dataSet = dataIn.dropRight(dataIn.length-batchSize)
      val Dsize = elemSize.toArray

      require(dataSet(0).length==Dsize(0)*Dsize(1),s"The number of data points per example in file does not equal ${Dsize(0)*Dsize(1)}")

      ////////////////Covert data to 2D Float Array
      var Fmat: Array[Array[Float]] = Array.empty
      if (areNums) {
        Fmat = dataSet.map(_.map(x => x.toFloat))
      } else {
        Fmat = dataSet.map(_.map(x => x.toList.head.toByte.toFloat))
      }
      val dataArr = Fmat.transpose

      //////////////put data into 2D Vector Array
      val dataVecArr: Array[Array[Vector]] = {
        var mat: Array[Array[Vector]] = Array.empty
        var vArr: Array[Vector] = Array.empty
        for (i <- 0 until Dsize(0)) {
          vArr = Array.empty
          for (j <- 0 until Dsize(1)) {
            vArr = vArr :+ new Vector(dataArr(i*Dsize(1)+j))
          }
          mat = mat :+ vArr
        }
        mat
      }

      val VField = new VectorField(Dsize(0),Dsize(1), (i, j) => dataVecArr(i)(j))
      val DF = DifferentiableField(VField, batchSize)

      val fieldOut = {
        if (areNums) {
          DF.forward
         }else {DF.forward / 255f}
        }
      fieldOut
    }
}
