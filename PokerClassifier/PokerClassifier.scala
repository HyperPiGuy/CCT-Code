/**
  * Created by Matt Fontaine on 8/17/2016.
  */

import java.io.File

import libcog._
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.policy.StandardLearningRule
import toolkit.neuralnetwork.source.CSVDataSource
import toolkit.neuralnetwork.util._
import toolkit.neuralnetwork.DifferentiableField
import cogio.fieldstate.FieldState

object PokerClassifier extends App {
  val dir = "PokerNet"
  val outputClasses = 2
  val lr = StandardLearningRule(0.01f, 0.9f, 0.0005f)

  var classFound: Float = 0
  val data = CSVDataSource(new File(dir,"myhand.csv").toString,1,header=false,areNums=true,Shape(5,2))
  val actuatorData = new Array[Float](2)

  def Net(className: String, elemFilter: Int) = new ComputeGraph{
    val data = CSVDataSource(new File(dir,"myhand.csv").toString,1,header=false,areNums=true,Shape(5,2))
    val b1 = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\b1.layer")).toField,1)
    val fc1 = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\fc1.layer")).toField,1)
    val fc2 = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\fc2.layer")).toField,1)
    val fc3 = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\fc3.layer")).toField,1)
    val fc4 = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\fc4.layer")).toField,1)
    val fc5 = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\fc5.layer")).toField,1)
    val fc6 = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\fc6.layer")).toField,1)
    val bf = DifferentiableField(FieldState.loadFromFile(new File(dir,className+"\\bf.layer")).toField,1)

    val tb1 = elemFilter match{
      case 0 => Bias(data, b1, true)
      case 1 => val noSuitVector = new VectorField(5,2,(i,j)=>if(j==0){Vector(1,(n)=>0)}else{Vector(1,(n)=>1)})
                val noSuitFilter = DifferentiableField(noSuitVector,1)
                Bias(data*noSuitFilter, b1, true)
      case 2 => val noFaceVector = new VectorField(5,2,(i,j)=>if(j==1){Vector(1,(n)=>0)}else{Vector(1,(n)=>1)})
                val noFaceFilter = DifferentiableField(noFaceVector,1)
                Bias(data*noFaceFilter, b1, true)
    }
    val tfc1 = FullyConnected(tb1,fc1)
    val tr1 = Tanh(tfc1)
    val tfc2 = FullyConnected(tr1,fc2)
    val tr2 = Tanh(tfc2)
    val tfc3 = FullyConnected(tr2,fc3)
    val tr3 = Tanh(tfc3)
    val tfc4 = FullyConnected(tr3,fc4)
    val tr4 = Tanh(tfc4)
    val tfc5 = FullyConnected(tr4,fc5)
    val tr5 = Tanh(tfc5)
    val tfc6 = FullyConnected(tr5,fc6)
    val tbf = Bias(tfc6,bf,true)
    val classification = Classify(tbf.forward,1,.001f)

    val classToSField = GPUOperator(ScalarField(2).fieldType) {
      _forEachTensorElement(classification.tensorShape) {
        val element = _tensorElementVar(classification)
        element := _readTensorElement(classification, _tensorElement)
        _writeTensorElement(_out0, element, _tensorElement, _tensorElement)
      }
    }
    Actuator(classToSField, actuatorData, (column) => column)
  }

  val FourNet = Net("Four",1)
  FourNet withRelease{
    FourNet.step
    classFound =  actuatorData(1)
    if(classFound==1) println("Its a Four of a Kind!")
  }

  if(classFound==0){
    val HouseNet = Net("House",1)
    HouseNet withRelease{
      HouseNet.step
      classFound =  actuatorData(1)
      if(classFound==1) println("Its a Full House!")
    }
  }

  if(classFound==0){
    val FlushNet = Net("Flush",2)
    FlushNet withRelease{
      FlushNet.step
      classFound =  actuatorData(1)
    }
    if(classFound==1){
      val stFlushNet = Net("Straight",1)
      stFlushNet withRelease{
        stFlushNet.step
        classFound = actuatorData(1)+1
      }
    }
    if(classFound==1) println("Its a Flush!")
    if(classFound==2) println("Its a Straight Flush!")
  }

  if(classFound==0){
    val StraightNet = Net("Straight",1)
    StraightNet withRelease{
      StraightNet.step
      classFound =  actuatorData(1)
      if(classFound==1) println("Its a Straight!")
    }
  }

  if(classFound==0){
    val ThreeNet = Net("Three",1)
    ThreeNet withRelease{
      ThreeNet.step
      classFound =  actuatorData(1)
      if(classFound==1) println("Its a Three of a Kind!")
    }
  }

  if(classFound==0){
    val TwoNet = Net("Two",1)
    TwoNet withRelease{
      TwoNet.step
      classFound =  actuatorData(1)
      if(classFound==1) println("Its a Two Pair!")
    }
  }

  if(classFound==0){
    val OneNet = Net("One",1)
    OneNet withRelease{
      OneNet.step
      classFound =  actuatorData(1)
      if(classFound==1) println("Its a Pair!")
    }
  }

  if(classFound==0){
    println("High Card")
  }
}
