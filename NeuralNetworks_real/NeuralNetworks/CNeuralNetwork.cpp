#include "CNeuralNetwork.h"
#include "ActivationFunctions.h"
#include "Log.h"

#include <fstream>
#include <direct.h>
#include <ctime>
#include <filesystem>

bool CURR_IS_POS = true;

//neural networks model with entire neurons data
vector<vector<CNeuron>> NN_MODEL;

string datetime()
{
	time_t rawtime;
	struct tm* timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, 80, "Date_%d_%m_%Y_Time_%H_%M_%S", timeinfo);
	return string(buffer);
}

double randd() {
	double retVal = (double)rand() / ((double)RAND_MAX + 1);
	retVal = retVal * 0.01;

	return retVal;
}

CNeuron::CNeuron(int iLayerIdx, int iNeuronIdx, int iSize, double iWeightDistributionFact)
{
	_layerIdx = iLayerIdx;
	_neuronIdx = iNeuronIdx;

	_bias = 0.1*iWeightDistributionFact;
	_output = 0;
	_input = 0;
	_error = 0;

	_fired = true;

	_weights.resize(iSize);
	for (int idx = 0; idx < iSize; idx++)
		_weights[idx] = randd()*iWeightDistributionFact;
}

void CNeuron::ComputeOutput()
{
	int prevLayerIdx = _layerIdx - 1;

	if (prevLayerIdx >= 0)//if not an input layer
	{
		int neuronsCount = NN_MODEL[prevLayerIdx].size();// for previous layer neuron count

		//      x1
		//		  \
		//		   \ w1
		//		    \
		//		     \
		// x2---w2--- N ------  + bias ----> Activation function --- > output
		//			 /
		//			/
		//		   /w3
		//		  /
		//		x3
		//
		// S = (w1x1 + w2x2 + w3x3 + w4x4 + ...) + bias 
		// output = ActivationFunction(S)
		//

		double sigmaXiWi = 0;
		for (int neuronIdx = 0; neuronIdx < neuronsCount; neuronIdx++)
		{
			CNeuron neuron = NN_MODEL[prevLayerIdx][neuronIdx];

			double x = neuron._output;
			double weight = _weights[neuronIdx];

			sigmaXiWi = sigmaXiWi + x * weight;//S = w1x1 + w2x2 + w3x3 + w4x4 + ...
		}

		_input = sigmaXiWi + _bias; // S = (w1x1 + w2x2 + w3x3 + w4x4 + ...) + bias 
		_fired = CActivationFunction::ActivationFunctions(_input, _output);
	}
}

void CNeuron::computeError()
{
	int layerCount = NN_MODEL.size();
	if (_layerIdx == layerCount - 1)// for output layer
	{
		double actual = CURR_IS_POS ? 1 : 0;
		_error = actual - _output;
		//_error = _error * _error*0.5; // cost function = half of sum of sq of errors 
	}
	else
	{
		_error = 0;

		// taking errors of the neurons in the next layer
		// and multiplying them with weight of corresponding connection between
		// next layer neuron and and the current neuron
		//				 e1
		//				/
		//          w1 /
		//            /    
		//           /	    w2
		// current  N ---------  e2
		//           \
		//            \
		//          w3 \
		//				\
		//				 e3
		//
		// Error of N is 
		// E = w1e1 + w2e2 + w3e3 + ...

		vector<CNeuron> &nextLayer = NN_MODEL[_layerIdx + 1];
		for (int neuronIdx = 0; neuronIdx < nextLayer.size(); neuronIdx++)
		{
			CNeuron &neuron = nextLayer[neuronIdx];
			double weight = neuron._weights[_neuronIdx];
			double error = neuron._error;
			_error += weight * error;
		}

		//_error = _error / nextLayer.size();
	}
}

CNeuralNetwork::CNeuralNetwork(const vector<vector<double>> &iPosInputData,
	const vector<vector<double>> &iNegInputData,
	vector<int> iNeuronsCounts,
	double iLearningRate,
	double iReqAccuracy)
{
	_learningRate = iLearningRate;
	_neuronsCountInEachLayer = iNeuronsCounts;
	_posInputData = iPosInputData;
	_negInputData = iNegInputData;
	_acceptedError = 1 - iReqAccuracy;

	InitializeNN();
}

CNeuralNetwork::CNeuralNetwork(string iStrModelPath)
{
	bool stat = false;

	stat = LoadBasicInfoOfNN(iStrModelPath);

	if (stat == true)
	{
		SetConsoleTextColor("Green");
		cout << "Basic file loaded succesfully" << endl;
	}
	else
	{
		SetConsoleTextColor("Red");
		cout << "Basic file loading failed" << endl;
	}

	SetConsoleTextColor("White");

	stat = LoadModelDataOfNN(iStrModelPath);

	if (stat == true)
	{
		SetConsoleTextColor("Green");
		cout << "model loaded succesfully" << endl;
	}
	else
	{
		SetConsoleTextColor("Red");
		cout << "model loading failed" << endl;
	}

	SetConsoleTextColor("White");
}

bool CNeuralNetwork::LoadBasicInfoOfNN(string iStrModelPath)
{
	try
	{
		string strBasicInfoFilePath = iStrModelPath + "\\BasicInfo.txt";

		fstream basicInfofile;
		basicInfofile.open(strBasicInfoFilePath.c_str());

		string strLine = "";
		getline(basicInfofile, strLine);
		if (strLine != "learning rate")
			return false;

		getline(basicInfofile, strLine);
		_learningRate = stod(strLine);

		getline(basicInfofile, strLine);
		if (strLine != "accepted error")
			return false;

		getline(basicInfofile, strLine);
		_acceptedError = stod(strLine);

		getline(basicInfofile, strLine);
		if (strLine != "number of layers")
			return false;

		getline(basicInfofile, strLine);
		_numOfLayers = stod(strLine);

		getline(basicInfofile, strLine);
		if (strLine != "neurons count in each layer")
			return false;

		NN_MODEL.clear();
		_neuronsCountInEachLayer.clear();

		NN_MODEL.resize(_numOfLayers);
		_neuronsCountInEachLayer.resize(_numOfLayers);
		for (int idx = 0; idx < _numOfLayers; idx++)
		{
			getline(basicInfofile, strLine);
			int neuronCount = stoi(strLine);

			_neuronsCountInEachLayer[idx] = neuronCount;
			NN_MODEL[idx].resize(neuronCount);
		}

		basicInfofile.close();
	}
	catch (...)
	{
		return false;
	}

	return true;
}

bool CNeuralNetwork::LoadModelDataOfNN(string iStrModelPath)
{
	try
	{
		for (int layerIdx = 0; layerIdx < _numOfLayers; layerIdx++)
		{
			vector<CNeuron> &currLayer = NN_MODEL[layerIdx];
			string strLayerFile = iStrModelPath + "\\Layer" + to_string(layerIdx) + ".txt";

			int weightsCount = layerIdx == 0 ? 1 : _neuronsCountInEachLayer[layerIdx - 1];

			fstream LayerInfofile;
			LayerInfofile.open(strLayerFile.c_str());

			string strLine = "";
			getline(LayerInfofile, strLine);

			while (LayerInfofile.eof() == false)
			{
				if (strLine.find("neuron") == -1)
					return false;

				string str2 = "neuron ";
				string str3 = "";
				strLine.replace(strLine.find(str2), str2.length(), str3);

				int neuronIdx = stoi(strLine);

				CNeuron &neuron = currLayer[neuronIdx];

				neuron._layerIdx = layerIdx;
				neuron._neuronIdx = neuronIdx;

				getline(LayerInfofile, strLine);
				if (strLine != "bias")
					return false;

				getline(LayerInfofile, strLine);
				istringstream biasStream(strLine);
				double bias;
				biasStream >> bias;
				neuron._bias = bias;

				getline(LayerInfofile, strLine);
				if (strLine != "output")
					return false;

				getline(LayerInfofile, strLine);
				istringstream outputStream(strLine);
				double output;
				outputStream >> output;
				neuron._output = output;

				getline(LayerInfofile, strLine);
				if (strLine != "input")
					return false;

				getline(LayerInfofile, strLine);
				istringstream inputStream(strLine);
				double input;
				inputStream >> input;
				neuron._input = input;

				getline(LayerInfofile, strLine);
				if (strLine != "error")
					return false;

				getline(LayerInfofile, strLine);
				istringstream errorStream(strLine);
				double error;
				errorStream >> error;
				neuron._error = error;

				getline(LayerInfofile, strLine);
				if (strLine != "fired")
					return false;

				getline(LayerInfofile, strLine);
				istringstream firedStream(strLine);
				bool fired;
				firedStream >> fired;
				neuron._fired = fired;

				getline(LayerInfofile, strLine);
				if (strLine != "weights")
					return false;

				neuron._weights.resize(weightsCount);
				for (int weightIdx = 0; weightIdx < weightsCount; weightIdx++)
				{
					getline(LayerInfofile, strLine);
					istringstream weightStream(strLine);
					double weight;
					weightStream >> weight;
					neuron._weights[weightIdx] = weight;
				}

				getline(LayerInfofile, strLine);
			}

			LayerInfofile.close();
		}
	}
	catch (...)
	{
		return false;
	}

	return true;
}

void CNeuralNetwork::SetTrainingData(const vector<vector<double>> &iPosInputData, const vector<vector<double>> &iNegInputData)
{
	_posInputData = iPosInputData;
	_negInputData = iNegInputData;
}

void CNeuralNetwork::InitializeNN()
{
	auto startItr = _neuronsCountInEachLayer.begin();
	_neuronsCountInEachLayer.insert(startItr, _posInputData[0].size()); //first layer with neurons of input data
	_neuronsCountInEachLayer.push_back(1); //last layer with one neuron i.e., output

	_numOfLayers = _neuronsCountInEachLayer.size();//input + hidden + output layer

	//assign input data to the first layer
	NN_MODEL.clear();
	NN_MODEL.resize(_numOfLayers);
	for (int layerIdx = 0; layerIdx < _numOfLayers; layerIdx++)
	{
		int neuronCount = _neuronsCountInEachLayer[layerIdx];

		vector<CNeuron> &layer = NN_MODEL[layerIdx];
		layer.clear();
		layer.resize(neuronCount);
		int size = layerIdx == 0 ? 1 : _neuronsCountInEachLayer[layerIdx - 1];
		int sizeOfChildLayer = layerIdx < _numOfLayers - 1 ? _neuronsCountInEachLayer[layerIdx + 1] : 1;

		//Normalized Xavier/Glorot Initialization
		double weightDistributionFact = sqrt(6.0 / (size + sizeOfChildLayer));

		for (int neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
		{
			CNeuron &neuron = layer[neuronIdx];
			neuron = CNeuron(layerIdx, neuronIdx, size, weightDistributionFact);

			if (layerIdx == 0)//first layer is input
			{
				neuron._input = _posInputData[0][neuronIdx];//take first positive data and initialize
				neuron._output = _posInputData[0][neuronIdx];//take first positive data and initialize
				neuron._bias = 0;
			}

			//if (layerIdx == _numOfLayers - 1)//last layer will have no bias
			//	neuron._bias = 0;
		}
	}
}

void CNeuralNetwork::StartTraining()
{
	int posInputDataCount = _posInputData.size();
	int negInputDataCount = _negInputData.size();

	int posDataIdx = 0, negDataIdx = 0;
	int totalInputData = posInputDataCount + negInputDataCount;
	int posLeanredDatCount = 0, negLeanredDatCount = 0;

	//0 - not learned,
	//1 - learnt
	vector<int> posDataStat(posInputDataCount, 0);
	vector<int> negDataStat(negInputDataCount, 0);

	// 0 - all data to be iterated,
	// 1 - only not learned data to be iterated
	//
	//start with 0 i.e., all input data
	//sfter some iteration we can decide if model has to train for only that are done yet
	//by this we can save time
	int typeOfTraining = 0;

	//in training pos and neg data is fed alternativly and trained
	for (int itr = 0; itr < _posInputData.size(); itr++)
	{
		// when itr is odd number positive data is given and 
		// when itr is even number negetive data is given 
		CURR_IS_POS = itr % 2 == 1 ? true : false;

		int dataInputIdx = 0;
		if (CURR_IS_POS == true)
		{
			posDataIdx = posDataIdx % posInputDataCount;
			dataInputIdx = posDataIdx;
			posDataIdx++;
		}
		else
		{
			negDataIdx = negDataIdx % negInputDataCount;
			dataInputIdx = negDataIdx;
			negDataIdx++;
		}

		int &inputDataStat = CURR_IS_POS == true ? posDataStat[dataInputIdx] : negDataStat[dataInputIdx];

		//skip this input data in case already traine to accuracy
		//when type 1 is choosen
		if (typeOfTraining == 1 && inputDataStat == 1 && itr != 4999)
			continue;

		string strDataType = CURR_IS_POS == true ? " pos Data " : " neg Data ";
		cout << endl << itr << " iteration Porcessing Data - " << dataInputIdx << strDataType << endl;

		const vector<double> &inputData = CURR_IS_POS == true ? _posInputData[dataInputIdx] : _negInputData[dataInputIdx];

		//set input layer data
		SetInputLayerData(inputData);

		//move forward from layer 1 to end
		PropagateForwardFrom(1);

		//perform back propagation
		//compute error in each neuron and udate weights
		ComputeErrorsByBackPropUpTo(1);
		for (int layerIdx = 0; layerIdx < _numOfLayers; layerIdx++)
			UpdateWeights(layerIdx);

		double error = GetError();
		double output = GetOutput();
		string strConsoleColor = abs(error) < _acceptedError ? "Green" : "Red";
		inputDataStat = abs(error) < _acceptedError ? 1 : 0;

		SetConsoleTextColor(strConsoleColor);//set color
		cout << "error = " << error << endl;
		cout << "output = " << output << endl;
		SetConsoleTextColor("White");// reset the color


		//exit if all the pos and neg data is learnt by the model
		if (abs(error) > _acceptedError)
		{
			posLeanredDatCount = 0;
			negLeanredDatCount = 0;
		}
		else
		{
			if (CURR_IS_POS == true)
				posLeanredDatCount++;
			else
				negLeanredDatCount++;

			if (posLeanredDatCount > posInputDataCount && negLeanredDatCount > negInputDataCount)
			{
				cout << "Required Accuracy is achieved for all input data" << endl;
				break;
			}
		}

		if (itr == 4999)
		{
			SetConsoleTextColor("Blue");
			cout << "\n 5000 iteration comlpeted \n do you want to run another 5000\n" << endl;
			cout << "enter 0 to iterate all the input data again" << endl;
			cout << "enter 1 to iterate data that not reached to reuired accuracy in training" << endl;
			cout << "enter other number to end the training" << endl;
			
			int input = 0;
			cin >> input;

			if (input == 0) //all
				typeOfTraining = 0;
			else if (input == 1)//only train that are not accurate
				typeOfTraining = 1;
			else
				break;

			itr = -1;// for another loop

			SetConsoleTextColor("White");
		}
	}

	cout << "Training completed" << endl;

	SaveNeuralNetwork();
}

double CNeuralNetwork::GetError()
{
	return NN_MODEL[_numOfLayers - 1][0]._error;
}

double CNeuralNetwork::GetOutput()
{
	return NN_MODEL[_numOfLayers - 1][0]._output;
}

void CNeuralNetwork::ComputeErrorsByBackPropUpTo(int iLayerIdx)
{
	//start calculating errors from output layer to iLayerIdx in back direction
	for (int layerIdx = _numOfLayers - 1; layerIdx >= iLayerIdx; layerIdx--)
	{
		vector<CNeuron> &layer = NN_MODEL[layerIdx];
		for (int neuronIdx = 0; neuronIdx < layer.size(); neuronIdx++)
		{
			CNeuron &neuron = layer[neuronIdx];
			neuron.computeError();
		}
	}
}

void CNeuralNetwork::UpdateWeights(int iLayerIdx)
{
	if (iLayerIdx <= 0)
		return;

	//updates weights of neurons in current layer according to the error
	vector<CNeuron> &currLayer = NN_MODEL[iLayerIdx];
	int connectedLayerIdx = iLayerIdx - 1;//previous layer or parent layer of current layer

	for (int neuronIdx = 0; neuronIdx < currLayer.size(); neuronIdx++)
	{
		CNeuron &neuron = currLayer[neuronIdx];
		double deriv = CActivationFunction::DerivativeActivationFunctions(neuron._input);
		double error = neuron._error;
		double &bias = neuron._bias;
		vector<double> &weights = neuron._weights;

		for (int weightIdx = 0; weightIdx < weights.size(); weightIdx++)
		{
			const CNeuron &connectedWeightNeuron = NN_MODEL[connectedLayerIdx][weightIdx];
			double input = connectedWeightNeuron._output;

			//SGD
			weights[weightIdx] += (_learningRate * error * deriv * input);
		}

		//bias -= (_learningRate * error * deriv );
	}
}

//update values neuron by neuron and layer by layer
//updating from current layer to output layer
void CNeuralNetwork::PropagateForwardFrom(int iFromLayerIdx)
{
	if (iFromLayerIdx <= 0)
		return;

	for (int layIdx = iFromLayerIdx; layIdx < _numOfLayers; layIdx++)
	{
		vector<CNeuron> &layer = NN_MODEL[layIdx];
		int neuronCount = layer.size();

		for (int neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
		{
			CNeuron &neuron = layer[neuronIdx];
			neuron.ComputeOutput();
		}
	}
}

void CNeuralNetwork::PropagteBackWardsTo(int iToLayerIdx)
{

}

void CNeuralNetwork::SaveNeuralNetwork()
{
	try
	{
		string strFolderName = datetime();
		string strFolder = string("E:\\PSV\\NeuralNetworks\\NN_") + strFolderName;

		//create folder to store the data
		_mkdir(strFolder.c_str());

		// Creation of ofstream class object
		ofstream basicFile;

		string strBasicInfoFile = strFolder + string("\\BasicInfo.txt");
		basicFile.open(strBasicInfoFile.c_str());

		basicFile << "learning rate" << endl << _learningRate << endl;
		basicFile << "accepted error" << endl << _acceptedError << endl;
		basicFile << "number of layers" << endl << _numOfLayers << endl;
		basicFile << "neurons count in each layer" << endl;

		//save neurons count in each layer
		for (int idx = 0; idx < _numOfLayers; idx++)
			basicFile << _neuronsCountInEachLayer[idx] << endl;

		basicFile.close();

		// save model data
		for (int layerIdx = 0; layerIdx < _numOfLayers; layerIdx++)
		{
			string strFile = "\\Layer" + std::to_string(layerIdx);
			string strLayerFile = strFolder + strFile;

			ofstream layerOutput;
			layerOutput.open(strLayerFile.c_str());

			vector<CNeuron> &layer = NN_MODEL[layerIdx];
			int neuronCount = _neuronsCountInEachLayer[layerIdx];
			for (int neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
			{
				CNeuron &neuron = layer[neuronIdx];

				layerOutput << "neuron " << neuronIdx << endl;
				layerOutput << "bias" << endl << neuron._bias << endl;
				layerOutput << "output" << endl << neuron._output << endl;
				layerOutput << "input" << endl << neuron._input << endl;
				layerOutput << "error" << endl << neuron._error << endl;
				layerOutput << "fired" << endl << neuron._fired << endl;

				layerOutput << "weights" << endl;
				for (int weightIdx = 0; weightIdx < neuron._weights.size(); weightIdx++)
					layerOutput << neuron._weights[weightIdx] << endl;
			}

			layerOutput.close();
		}
	}
	catch (...)
	{
		cout << "Neural Network model saving failed"<<endl;
		return;
	}

	cout << "Neural Network model saved successfully" << endl;
}

void CNeuralNetwork::SetInputLayerData(const vector<double> &iInputData)
{
	vector<CNeuron> &inputLayer = NN_MODEL[0]; //input layer

	for (int inputIdx = 0; inputIdx < iInputData.size(); inputIdx++)
	{
		CNeuron &neuron = inputLayer[inputIdx];
		neuron._input = iInputData[inputIdx];
		neuron._output = iInputData[inputIdx];
	}
}
