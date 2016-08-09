package toolkit.neuralnetwork.util

import libcog._

object Confusion{

  /** Create a Confusion matrix given the test and known labels
    * The matrix illustrates what classifications errors the network is making
    *
    * The matrix is read by:
    * Coordinate (x,y) = number of test examples classified x when correct label was y
    * Top left is (0,0)
    * Bottom right is (classes,classes)
    *
    *
    * @param testLabels labels that the network has made
    * @param correctLabels known labels from the dataset
    * @param batchSize number of examples in field
    * @param classes number of possible classifications of data
    * @return A ScalarField of the classifications versus correct classifications
    */
  def apply (testLabels: Field,
             correctLabels: Field,
             batchSize: Int,
             classes: Int): Field = {

    val length = batchSize*classes

    def makeScalar(v: VectorField): ScalarField = {
      val scalartype = ScalarField(length)
      val sField = GPUOperator(scalartype.fieldType) {
        _forEachTensorElement(v.tensorShape) {
          val element = _tensorElementVar(v)
          element := _readTensorElement(v, _tensorElement)
          _writeTensorElement(_out0, element, _tensorElement, _tensorElement)
        }
      }
      sField
    }

    val totalScalar = makeScalar(testLabels)+2f*makeScalar(correctLabels)

    val confField = GPUOperator(ScalarField(classes,classes).fieldType){
      val i = _intVar()
      val j = _intVar()
      val x = _intVar()
      val y = _intVar()
      val num = _intVar()
      num := 0
      _for(i:=0,i<length,i+=classes){
        x:=0
        y:=0
        _for(j:=0,j<classes,j+=1){
          val d = _readTensorElement(totalScalar,i+j,_tensorElement)
          _if(_row===_column){
            _if(j===_row && d===3){
              num+=1
            }
          }
          _else {
            _if(j === _row && d === 1) {
              y:=1
            }
            _elseif(j === _column && d === 2) {
              x:=1
            }
            _if(x===1&&y===1){
              num+=1
            }
          }
        }
      }
      _writeTensorElement(_out0,num,_row,_column,_tensorElement)
    }
    confField
  }
}