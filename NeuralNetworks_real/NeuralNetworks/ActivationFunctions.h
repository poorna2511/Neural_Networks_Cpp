#pragma once

#include <iostream>
#include <complex>

using namespace std;

class CActivationFunction {

public:
	CActivationFunction() {}

	static bool ActivationFunctions(double iInput, double &oOutput);
	static double DerivativeActivationFunctions(double iInput);
	
	static bool FastSigmoid(double iInput, double &oOutput);

	static bool Tanh(double iInput, double &oOutput);
	static double DerivativeTanh(double iInput);

	static bool ReLU(double iInput, double &oOutput);
	static double DerivativeReLU(double iInput);
};
