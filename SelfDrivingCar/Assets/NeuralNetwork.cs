using System.Collections.Generic;
using System;
using Random = UnityEngine.Random;
using UnityEngine;

namespace NEA
{
	public class NeuralNetwork:MonoBehaviour
	{
		public float[,] IToH1Weights = new float[4, 4];
		public float[,] H1ToH2Weights = new float[4, 4];
		public float[,] H2ToOutputWeights = new float[2, 4];
		public float[] Biases = new float[4];

		public List<float> hiddenLayer1;
		public List<float> hiddenLayer2;
		public List<float> outputLayer;
		public float[] finalOutputs;

		public void InitialiseRandomNeuralNetwork()
		{
			RandomiseWeights(IToH1Weights);
			RandomiseWeights(H1ToH2Weights);
			RandomiseWeights(H2ToOutputWeights);
			RandomiseBiases();
		}

		public float[] RunNeuralNetwork(float[] Inputs)
		{
			finalOutputs = new float[2]; //Final output contains acceleration value and turning value
			hiddenLayer1 = Forward(Inputs, IToH1Weights); // Output of the input layer
			hiddenLayer2 = Forward(hiddenLayer1.ToArray(), H1ToH2Weights); //Output of the first hidden layer
			outputLayer = Forward(hiddenLayer2.ToArray(), H2ToOutputWeights); //Output of second hidden layer
			//Use sigmoid on first value to get value between 0 and 1 so that no negative acceleration occurs
			float acceleration = Sigmoid(outputLayer[0]);
			//Use tanh on second value to get value between -1 and 1 so that car turns left and right
			float turnFactor = (float)Math.Tanh(outputLayer[1]);
			finalOutputs[0] = acceleration;
			finalOutputs[1] = turnFactor;
			return finalOutputs;
		}

		public void RandomiseBiases()
		{
			for (int i = 0; i < Biases.Length; i++)
			{
				Biases[i] = Random.Range(-1f, 1f);
			}
		}

		public void RandomiseWeights(float[,] Weights)
		{
			for (int i = 0; i < Weights.GetLength(0); i++)
			{
				for (int j = 0; j < Weights.GetLength(1); j++)
				{
					Weights[i, j] = Random.Range(-1f, 1f);
				}
			}
		}

		public List<float> Forward(float[] inputs, float[,] Weights)
		{
			//Output list represents outputs of an entire layer
			List<float> outputs = new List<float>();
			for (int i = 0; i < Weights.GetLength(0); i++)
			{
				float output = 0;
				for (int j = 0; j < Weights.GetLength(1); j++)
				{
					//Calculate the dot product of an inputs and weights
					output += inputs[j] * Weights[i, j];
				}
				// Add on bias to output of a neuron and apply tanh activation funciton
				outputs.Add((float)Math.Tanh(output + Biases[i])); 
			}
			return outputs;
		}

		public float Sigmoid(float input)
		{
			return (float)(1 / (1 + Math.Exp(-input)));
		}
	}
}
