using UnityEngine;
using TMPro;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Unity.VisualScripting;

namespace NEA
{
	public class NaturalSelection : MonoBehaviour
	{
		public bool solutionFound = false;
		public TextMeshProUGUI generationText;
		public List<CarController> bestCars;
		public List<CarController> carControllers;

		[SerializeField] Transform startingPos;
		[SerializeField] GameObject car;
		[SerializeField] Vector3 startingRot;

		[SerializeField] float parameterTweakValue = 0.25f;
		[SerializeField] int initialPopulation = 100;
		[SerializeField] float mutationProbablility = 0.01f;
		[SerializeField] float crossoverRate = 0.5f;
		[SerializeField] int numOfBestCarsToPick = 50;
		[SerializeField] bool reset = false;

		[SerializeField] List<CarController> aliveCars;

		private CarController testCar;
		private int generation = 1;
		private int dead = 0;


		private void Start()
		{
			solutionFound = (PlayerPrefs.GetInt("solutionFound") != 0); //Set solutionFound to the value in playerprefs
			if (solutionFound)
			{
				testCar = Instantiate(car, startingPos.position, Quaternion.Euler(startingRot)).GetComponent<CarController>();
				testCar.solutionFound = true;
			}
		}

		public void Restart()
		{
			
			foreach (CarController carController in carControllers)
			{
				Destroy(carController.gameObject);
			}
			dead = 0;
			generation = 1;
			PlayerPrefs.SetInt("solutionFound", 0);
			solutionFound = false;
			generationText.text = "Generation: " + generation.ToString();
			SpawnCars();
			carControllers = FindObjectsOfType<CarController>().ToList();
			carControllers = SortCarsByDistanceTravelled(carControllers);
		}

		private void SpawnCars()
		{
			for (int i = 0; i < initialPopulation; i++)
			{
				Instantiate(car, startingPos.position, Quaternion.Euler(startingRot));
			}
		}

		private void Update()
		{
			if (!solutionFound)
			{
				carControllers = FindObjectsOfType<CarController>().ToList();
				carControllers = SortCarsByDistanceTravelled(carControllers);
				CheckIfAllCarsDead();
				CheckIfSolutionFound();
				SetCameraToBestCar();
			}
		}

		private void SetCameraToBestCar()
		{
			aliveCars = new List<CarController>();
			for (int i = 0; i < carControllers.Count; i++)
			{
				if (!carControllers[i].isDead)
				{
					aliveCars.Add(carControllers[i]);
				}
			}
			aliveCars.Reverse();
			aliveCars[0].camera.SetActive(true);
			aliveCars[0].camera.GetComponent<Camera>().enabled = true;
			aliveCars[0].camera.GetComponent<AudioListener>().enabled = true;
			for (int i = 1; i < aliveCars.Count; i++)
			{
				aliveCars[i].camera.SetActive(false);
				aliveCars[i].camera.GetComponent<Camera>().enabled = false;
				aliveCars[i].camera.GetComponent<AudioListener>().enabled = false;
			}
		}

		private void CheckIfSolutionFound()
		{
			//Check every car to see whether the solutionFound bool is true
			for (int i = 0; i < carControllers.Count; i++)
			{
				if(carControllers[i].solutionFound == true) 
				{
					generationText.text = "PATH FOUND. TRAINING COMPLETE";
					PlayerPrefs.SetInt("solutionFound", 1);
					//Set the playerpref value of solutionfFound to 1 if sol
					for (int j = 0; j < carControllers.Count; j++)
					{
						//Destroy all cars other than the one that found the solution
						if(carControllers[j] != carControllers[i])
						{
							print("Destroying");
							Destroy(carControllers[j].gameObject);
						}
					}
					break;
				}
			}
		}

		private void SelectBestCars()
		{
			bestCars = new List<CarController>();
			carControllers.Reverse();
			for (int i = 0; i < numOfBestCarsToPick; i++)
			{
				bestCars.Add(carControllers[i]);
			}
		}

		private void CheckIfAllCarsDead()
		{
			dead = 0;
			for (int i = 0; i < carControllers.Count; i++)
			{
				if (carControllers[i].isDead)
				{
					dead++;
				}
			}
			if(dead >= carControllers.Count)
			{
				SelectBestCars();
				generation++;
				generationText.text = "Generation: " + generation.ToString();
				Repopulate();
				Crossover();
				Mutate();
			}
		}

		private void Repopulate()
		{
			/*
				Destroy every car on the map and then spawn in the next generation of cars
				based of the old generation's best performing cars
			 */
			for (int i = 0; i < carControllers.Count; i++)
			{
				Destroy(carControllers[i].gameObject);
			}
			for (int i = 0; i < initialPopulation; i++)
			{
				int randomIndex = Random.Range(0, bestCars.Count);
				var car = Instantiate(bestCars[randomIndex], startingPos.position, Quaternion.Euler(startingRot));
				var carController = car.GetComponent<CarController>();
				carController.enabled = true;
				carController.isFirstGen = false;
				carController.isDead = false;
				carController.transform.position = startingPos.position;
				carController.transform.eulerAngles = new Vector3(0f, 90f, 0f);
				carController.distanceTravelled = 0;
				carController.laps = 0;
				TweakNeuralNetwork(carController.neuralNetwork, bestCars[randomIndex].neuralNetwork);
			}
			carControllers = FindObjectsOfType<CarController>().ToList();
			Crossover();
			Mutate();
			dead = 0;
		}

		private void TweakNeuralNetwork(NeuralNetwork carController, NeuralNetwork goodCar)
		{
			/*
				This function offsets the weights and biases of the neural network of a good car
				and sets those new values to an underperforming car
			 */
			
			TweakParameter(carController.IToH1Weights, goodCar.IToH1Weights);
			TweakParameter(carController.H1ToH2Weights, goodCar.H1ToH2Weights);
			TweakParameter(carController.H2ToOutputWeights, goodCar.H2ToOutputWeights);

			for (int index = 0; index < carController.Biases.Length; index++)
			{
				float tweakValue = Random.Range(0f, parameterTweakValue);
				if (goodCar.Biases[index] + tweakValue <= 1 && goodCar.Biases[index] - tweakValue >= -1)
				{
					if (Random.Range(0f, 1f) <= 0.5f)
					{
						carController.Biases[index] = goodCar.Biases[index] + tweakValue;
					}
					else
					{
						carController.Biases[index] = goodCar.Biases[index] - tweakValue;
					}
				}
				else
				{
					tweakValue = Random.Range(0f, 0.1f);
					if (Random.Range(0f, 1f) <= 0.5f)
					{
						carController.Biases[index] = goodCar.Biases[index] + tweakValue;
					}
					else
					{
						carController.Biases[index] = goodCar.Biases[index] - tweakValue;
					}
				}
			}
		}

		private void TweakParameter(float[,] weights, float[,] goodWeights)
		{
			for (int row = 0; row < weights.GetLength(0); row++)
			{
				for (int column = 0; column < weights.GetLength(1); column++)
				{
					float tweakValue = Random.Range(0f, parameterTweakValue);
					if (goodWeights[row, column] + tweakValue <= 1 && goodWeights[row, column] - tweakValue >= -1)
					{
						if (Random.Range(0f, 1f) <= 0.5f)
						{
							weights[row, column] = goodWeights[row, column] + tweakValue;
						}
						else
						{
							weights[row, column] = goodWeights[row, column] - tweakValue;
						}
					}
					else
					{
						tweakValue = Random.Range(0f, 1f);
						if (Random.Range(0f, 1f) <= 0.5f)
						{
							weights[row, column] = goodWeights[row, column] + tweakValue;
						}
						else
						{
							weights[row, column] = goodWeights[row, column] - tweakValue;
						}
					}
				}
			}
		}

		private void Mutate()
		{
			/*
				This function has a small chance of mutating a random amount of cars each generation by
				Randomising one of its network parameters
			 */

			for (int i = 0; i < carControllers.Count; i++)
			{
				var car = carControllers[i];
				if (Random.Range(0f, 1f) < mutationProbablility)
				{
					int randomIndex = Random.Range(1, 5);
					switch (randomIndex)
					{
						case 1:
							car.neuralNetwork.RandomiseWeights(car.neuralNetwork.IToH1Weights);
							break;
						case 2:
							car.neuralNetwork.RandomiseWeights(car.neuralNetwork.H1ToH2Weights);
							break;
						case 3:
							car.neuralNetwork.RandomiseWeights(car.neuralNetwork.H2ToOutputWeights);
							break;
						case 4:
							car.neuralNetwork.RandomiseBiases();
							break;
					}
				}
			}

		}

		private void Crossover()
		{
			/*
			 
				This function loops through every car in the best cars array
				and selects 2 cars to be parents. If the parents aren't identical 
				then swap over weights or biases between parents to create a new child.
				This has a specific chance of happening (crossover rate)
			 
			 */
			for (int i = 0; i < bestCars.Count; i+=2)
			{
				try
				{
					var car1 = bestCars[i];
					var car2 = bestCars[i + 1];

					if(car1 != car2)
					{
						if (Random.Range(0f, 1f) < crossoverRate)
						{
							var temp = car1.neuralNetwork.IToH1Weights;
							car1.neuralNetwork.IToH1Weights = car2.neuralNetwork.IToH1Weights;
							car2.neuralNetwork.IToH1Weights = temp;
						}
						if (Random.Range(0f, 1f) < crossoverRate)
						{
							var temp = car1.neuralNetwork.H1ToH2Weights;
							car1.neuralNetwork.H1ToH2Weights = car2.neuralNetwork.H1ToH2Weights;
							car2.neuralNetwork.H1ToH2Weights = temp;
						}
						if (Random.Range(0f, 1f) < crossoverRate)
						{
							var temp = car1.neuralNetwork.H2ToOutputWeights;
							car1.neuralNetwork.H2ToOutputWeights = car2.neuralNetwork.H2ToOutputWeights;
							car2.neuralNetwork.H2ToOutputWeights = temp;
						}
						if (Random.Range(0f, 1f) < crossoverRate)
						{
							var temp = car1.neuralNetwork.Biases;
							car1.neuralNetwork.Biases = car2.neuralNetwork.Biases;
							car2.neuralNetwork.Biases = temp;
						}
					}
				}
				catch 
				{
					break;
				}
			}
		}

		private List<CarController> SortCarsByDistanceTravelled(List<CarController> cars)
		{
			//This function performs a merge sort on all cars by their distance travelled as shown in design section

			if(cars.Count <= 1)
			{
				return cars;
			}
			int midPoint = cars.Count / 2;
			List<CarController> leftSide = new List<CarController>();
			for (int i = 0; i < midPoint; i++)
			{
				leftSide.Add(cars[i]);
			}
			List<CarController> rightSide = new List<CarController>();
			for (int i = midPoint; i < cars.Count; i++)
			{
				rightSide.Add(cars[i]);
			}

			leftSide = SortCarsByDistanceTravelled(leftSide);
			rightSide = SortCarsByDistanceTravelled(rightSide);
			
			cars = Merge(cars, leftSide, rightSide);

			return cars;
		}

		private static List<CarController> Merge(List<CarController> cars, List<CarController> leftSide, List<CarController> rightSide)
		{
			int iteratorI = 0;
			int iteratorJ = 0;
			int iteratorK = 0;

			while (iteratorI < leftSide.Count && iteratorJ < rightSide.Count)
			{
				if (leftSide[iteratorI].distanceTravelled < rightSide[iteratorJ].distanceTravelled)
				{
					cars[iteratorK] = leftSide[iteratorI];
					iteratorI++;
				}
				else
				{
					cars[iteratorK] = rightSide[iteratorJ];
					iteratorJ++;
				}
				iteratorK++;
			}

			while (iteratorI < leftSide.Count)
			{
				cars[iteratorK] = leftSide[iteratorI];
				iteratorI++;
				iteratorK++;
			}

			while (iteratorJ < rightSide.Count)
			{
				cars[iteratorK] = rightSide[iteratorJ];
				iteratorJ++;
				iteratorK++;
			}

			return cars;
		}
	}
}
