#pragma once

#include <iostream>
#include <complex>
#include <vector>

using namespace std;

class CNeuron
{
public:
	double _bias;
	double _output;
	double _input;
	double _error;

	bool _fired;

	int _layerIdx;
	int _neuronIdx;

	vector<double> _weights;

	CNeuron() {}

	CNeuron(int iLayerIdx, int iNeuronIdx, int iSize, double iWeightDistributionFact);

	void ComputeOutput();

	void computeError();

};

class CNeuralNetwork
{
private:
	int _numOfLayers;

	double _learningRate;
	double _acceptedError;

	// neruons counts in hidden layers
	vector<int> _neuronsCountInEachLayer;

	//pos and neg data
	vector<vector<double>> _posInputData;
	vector<vector<double>> _negInputData;

public:
	CNeuralNetwork(const vector<vector<double>> &iPosInputData,
		const vector<vector<double>> &iNegInputData,
		vector<int> iNeuronsCounts,
		double iLearningRate, 
		double iReqAccuracy);

	CNeuralNetwork(string iStrModelPath);

	CNeuralNetwork() {}

	void SetTrainingData(const vector<vector<double>> &iPosInputData,
		const vector<vector<double>> &iNegInputData);

	bool LoadBasicInfoOfNN(string iStrModelPath);

	bool LoadModelDataOfNN(string iStrModelPath);
	   
	void InitializeNN();

	void StartTraining();

	void ComputeErrorsByBackPropUpTo(int iLayerIdx);

	void UpdateWeights(int iLayerIdx);

	void SetInputLayerData(const vector<double> &iInputData);

	void PropagateForwardFrom(int iLayerIdx);

	void PropagteBackWardsTo(int iLayerIdx);

	void SaveNeuralNetwork();

	double GetError();

	double GetOutput();
};
