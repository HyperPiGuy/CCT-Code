/**
  * Created by Matt Fontaine on 8/17/2016.
  */

import java.io.File
import libcog._
import toolkit.neuralnetwork.policy.StandardLearningRule
import toolkit.neuralnetwork.examples.networks.PokerNet
import cogio.fieldstate.FieldState

object PokerTrainer extends App {
  val dir = "C:\\HPE-CCT\\PokerClassifier\\resources"
  val outputClasses = 2
  val lr = StandardLearningRule(0.01f, 0.9f, 0.0005f)

  val netFileName = "Straight"

  val Net = new ComputeGraph {
    val nn = new PokerNet(new File(dir, "Straight.csv").toString, new File(dir, "StraightLabels.csv").toString, lr, 1, 600)
    probeAll
  }
  val NetSave = try {
    Net.step(7000)
    val b1 = FieldState.read(Net.read(Net.nn.b1.weights.forward))
    b1.saveToFile(new File(dir,netFileName+"\\b1.layer"))
    val fc1 = FieldState.read(Net.read(Net.nn.fc1.weights.forward))
    fc1.saveToFile(new File(dir,netFileName+"\\fc1.layer"))
    val fc2 = FieldState.read(Net.read(Net.nn.fc2.weights.forward))
    fc2.saveToFile(new File(dir,netFileName+"\\fc2.layer"))
    val fc3 = FieldState.read(Net.read(Net.nn.fc3.weights.forward))
    fc3.saveToFile(new File(dir,netFileName+"\\fc3.layer"))
    val fc4 = FieldState.read(Net.read(Net.nn.fc4.weights.forward))
    fc4.saveToFile(new File(dir,netFileName+"\\fc4.layer"))
    val fc5 = FieldState.read(Net.read(Net.nn.fc5.weights.forward))
    fc5.saveToFile(new File(dir,netFileName+"\\fc5.layer"))
    val fc6 = FieldState.read(Net.read(Net.nn.fc6.weights.forward))
    fc6.saveToFile(new File(dir,netFileName+"\\fc6.layer"))
    val bf = FieldState.read(Net.read(Net.nn.bf.weights.forward))
    bf.saveToFile(new File(dir,netFileName+"\\bf.layer"))
  }finally{
    Net.release
  }
}
