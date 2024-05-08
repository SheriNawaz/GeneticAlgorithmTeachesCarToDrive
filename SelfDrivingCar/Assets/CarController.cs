using UnityEngine;
using System.IO;

namespace NEA
{
    public class CarController : MonoBehaviour
    {
		public float[] accelerationAndTurnFactor;
		public NeuralNetwork neuralNetwork;
		public int laps = 0;
        public int generation = 0;
		public float distanceTravelled = 0;
        public bool isDead = false;
		public bool solutionFound = false;
		public bool isFirstGen = true;
		public string path = ".\\WeightsAndBiases.txt";
		public GameObject camera;

		[Header("Vehicle Parameters")]
        [SerializeField] float moveSpeed = 200f;
        [SerializeField] float turnSpeed = 90f;
        [SerializeField] Transform[] carPoints;

		private NaturalSelection naturalSelection;
        private Rigidbody rigidBody;
        private Vector3 startingPos;
        private Vector3 lastPosition;

		public float[] GetDistancesToWall()
        {
            float[] distances = new float[4];
			for (int i = 0; i < carPoints.Length; i++)
			{
                Ray ray = new Ray(carPoints[i].position, carPoints[i].forward);
                RaycastHit hit;

                if (Physics.Raycast(ray, out hit, 1000, LayerMask.GetMask("Walls")))
                {
                    Debug.DrawLine(ray.origin, hit.point, Color.red);
                    distances[i] = hit.distance;
                }
            }
            return distances;
        }

        public void Accelerate(float nnSpeed)
        {
            rigidBody.velocity = transform.forward * moveSpeed * nnSpeed;
        }

        public void Turn(float nnTurnFactor)
        {
            transform.eulerAngles += new Vector3(0, nnTurnFactor * turnSpeed, 0);
        }

        private void Start()
		{
			naturalSelection = FindObjectOfType<NaturalSelection>();
			solutionFound = naturalSelection.solutionFound;
			neuralNetwork = GetComponent<NeuralNetwork>();
			if (!solutionFound)
			{
				if (isFirstGen)
				{
					neuralNetwork.InitialiseRandomNeuralNetwork();
				}
			}
			else
			{
				print("Starting with knowledge");
				naturalSelection.generationText.text = "PATH FOUND. TRAINING COMPLETE";
				var learntValues = StartWithKnowledge();
				neuralNetwork.IToH1Weights = learntValues.Item1;
				neuralNetwork.H1ToH2Weights = learntValues.Item2;
				neuralNetwork.H2ToOutputWeights = learntValues.Item3;
				neuralNetwork.Biases = learntValues.Item4;
				
			}
			rigidBody = GetComponent<Rigidbody>();
            startingPos = transform.position;
        }

        private void Update()
		{
			ProcessNeuralNetwork();
			CheckIfSolutionFound();
		}

		private void CheckIfSolutionFound()
		{
			if (laps >= 3)
			{
				solutionFound = true;
				using (StreamWriter streamWriter = File.CreateText(path))
				{
					SaveData(streamWriter);
				}
			}
		}

		private void ProcessNeuralNetwork()
		{
			float[] inputs = GetDistancesToWall();
			if (!isDead)
			{
				accelerationAndTurnFactor = neuralNetwork.RunNeuralNetwork(inputs);
				Accelerate(accelerationAndTurnFactor[0]);
				Turn(accelerationAndTurnFactor[1]);
				distanceTravelled += Vector3.Distance(transform.position, lastPosition);
				lastPosition = transform.position;
			}
		}

		public (float[,], float[,], float[,], float[]) StartWithKnowledge()
		{
			float[,] IToH1Weights = new float[4, 4];
			float[,] H1ToH2Weights = new float[4, 4];
			float[,] H2ToOutputWeights = new float[2, 4];
			float[] Biases = new float[4];
			int increment = 0;
			foreach (string line in File.ReadLines(path))
			{
				switch (increment)
				{
					case 0: //First line of file contains contents of Input to hidden layer 1 weights
						ConvertLineTo2DArray(IToH1Weights, line);
						break;
					case 1: //Second line of file contains contents of hidden layer 1 to hidden layer 2 weights
						ConvertLineTo2DArray(H1ToH2Weights, line);
						break;
					case 2:  //Third line of file contains contents of hidden layer 2 to output layer  weights
						ConvertLineTo2DArray(H2ToOutputWeights, line);
						break;
					//Lines 4-7 contain the biases
					case 3: 
						Biases[0] = float.Parse(line);
						break;
					case 4:
						Biases[1] = float.Parse(line);
						break;
					case 5:
						Biases[2] = float.Parse(line);
						break;
					case 6:
						Biases[3] = float.Parse(line);
						break;
				}
				increment++;
			}
			return (IToH1Weights, H1ToH2Weights, H2ToOutputWeights, Biases);
		}

		private static void ConvertLineTo2DArray(float[,] emptyMatrix, string line)
		{
			//This function is necessary as every weight matrix is saved as a single line 
			string[] values = line.Split(); //Creat an array of values by splitting up the file at every space
			int valueCount = 0; //Iterator variable for the value array 
			for (int i = 0; i < emptyMatrix.GetLength(0); i++)
			{
				for (int j = 0; j < emptyMatrix.GetLength(1); j++)
				{
					emptyMatrix[i, j] = float.Parse(values[valueCount]);
					//iterate through the value array and set the next weight to the next value
					valueCount++;
				}
			}
		}

		public void SaveData(StreamWriter streamWriter)
		{
			SaveWeights(neuralNetwork.IToH1Weights, streamWriter);
			streamWriter.WriteLine();
			SaveWeights(neuralNetwork.H1ToH2Weights, streamWriter);
			streamWriter.WriteLine();
			SaveWeights(neuralNetwork.H2ToOutputWeights, streamWriter);
			streamWriter.WriteLine();
			SaveBiases(neuralNetwork.Biases, streamWriter);
		}

		private void SaveBiases(float[] biases, StreamWriter streamWriter)
		{
			for (int i = 0; i < biases.Length; i++)
			{
				streamWriter.WriteLine(biases[i]);
			}
		}

		private void SaveWeights(float[,] Weights, StreamWriter streamWriter)
		{
			for (int i = 0; i < Weights.GetLength(0); i++)
			{
				for (int j = 0; j < Weights.GetLength(1); j++)
				{
					streamWriter.Write(Weights[i, j] + " ");
				}
			}
		}


		private void OnCollisionEnter(Collision collision)
		{
			if (collision.gameObject.tag == "hazard")
			{
				try
				{
					isDead = true;
					rigidBody.velocity = Vector3.zero;
					rigidBody.angularVelocity = Vector3.zero;
					solutionFound = false;
					FindObjectOfType<NaturalSelection>().solutionFound = false;
					laps = 0;
				}
				catch
				{
					print("No Best Cars Found"); 
				}
				
			}
		}

		private void OnTriggerEnter(Collider other)
		{
			if (other.gameObject.tag == "lap")
			{
				laps++;
			}
		}
	}
}