package toolkit.neuralnetwork.examples.networks
/**
  * Created by Matthew Fontaine on 8/17/2016.
  */
  
import libcog._
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.policy.StandardLearningRule
import toolkit.neuralnetwork.util._
import toolkit.neuralnetwork.Implicits._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.layer.{BiasLayer, FullyConnectedLayer}
import toolkit.neuralnetwork.source.{CSVDataSource, CSVLabelSource}

class PokerNet(dataFile: String,
               labelFile: String,
               lr: StandardLearningRule,
               elemFilter: Int, //for elemFilter a 1 filters out the card suit, a 2 filters out the card face, 0 does nothing
               batchSize: Int){

  val data = CSVDataSource(dataFile, batchSize, header = false, areNums = true, Shape(5, 2))

  val b1 = elemFilter match{
    case 0 => BiasLayer(data, lr)
    case 1 => val noSuitVector = new VectorField(5,2,(i,j)=>if(j==0){Vector(batchSize,(n)=>0)}else{Vector(batchSize,(n)=>1)})
              val noSuitFilter = DifferentiableField(noSuitVector,batchSize)
              BiasLayer(data*noSuitFilter, lr)
    case 2 => val noFaceVector = new VectorField(5,2,(i,j)=>if(j==1){Vector(batchSize,(n)=>0)}else{Vector(batchSize,(n)=>1)})
              val noFaceFilter = DifferentiableField(noFaceVector,batchSize)
              BiasLayer(data*noFaceFilter, lr)
  }
  val fc1 = FullyConnectedLayer(b1, 100, lr)
  val r1 = Tanh(fc1)
  val fc2 = FullyConnectedLayer(r1, 50, lr)
  val r2 = Tanh(fc2)
  val fc3 = FullyConnectedLayer(r2, 20, lr)
  val r3 = Tanh(fc3)
  val fc4 = FullyConnectedLayer(r3, 10, lr)
  val r4 = Tanh(fc4)
  val fc5 = FullyConnectedLayer(r4, 5, lr)
  val r5 = Tanh(fc5)
  val fc6 = FullyConnectedLayer(r5, 2, lr)
  val bf = BiasLayer(fc6, lr)

  val label = CSVLabelSource(labelFile, batchSize, 2, header = false)
  val correct = CorrectCount(bf.forward, label.forward, batchSize, 0.01f) / batchSize
  val avgCorrect = NormalizedLowPass(correct, 0.001f)
  val loss = CrossEntropySoftmax(bf, label) / batchSize
  loss.activateSGD()
}
