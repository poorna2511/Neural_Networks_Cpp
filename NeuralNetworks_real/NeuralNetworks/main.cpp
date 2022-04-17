#include<iostream>
#include<vector>

#include "CNeuralNetwork.h"

using namespace std;

int main()
{
	vector<vector<double>> posData, negData;//just for sample we are taking empty vctor
	//after loading pos and neg data

	vector<int> neuronsCountsInHiddenLayer;
	neuronsCountsInHiddenLayer.push_back(100);
	neuronsCountsInHiddenLayer.push_back(100);
	neuronsCountsInHiddenLayer.push_back(100);

	double learningRate = 0.001;
	double requiredAccuracy = 0.70;

	CNeuralNetwork nnObj(posData, negData, neuronsCountsInHiddenLayer, learningRate, requiredAccuracy);
	nnObj.StartTraining();

	vector<double> testData;//fill
	//after loading testData

	nnObj.SetInputLayerData(testData);
	nnObj.PropagateForwardFrom(1);

	double output = nnObj.GetOutput();

	cout << "Output = " << output << endl;

	system("pause");
	return 0;
}
