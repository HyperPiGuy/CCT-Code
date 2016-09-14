package toolkit.neuralnetwork.source

import libcog._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.operator.indexToOneHotCode

/**
  * This is the label version of the CSV source
  * Place all of the labels into one single column
  *
  * @param fileName CSV file to load
  * @param batchSize Number of examples to read from file
  * @param classes Number of classifications
  * @param header true if there is a header row at top of file, false otherwise
  */
case class CSVLabelSource (fileName:String,
                           override val batchSize:Int,
                           classes:Int,
                           header:Boolean) extends DifferentiableField {
  override val forward: Field =
    indexToOneHotCode(CSVDataSource(fileName,batchSize,header,true,Shape(1,1)).forward,classes).toVectorField(0).toVectorField(0)
}
