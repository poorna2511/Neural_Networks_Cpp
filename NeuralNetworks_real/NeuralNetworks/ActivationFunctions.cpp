#include "ActivationFunctions.h"

bool CActivationFunction::ActivationFunctions(double iInput, double &oOutput)
{
	return Tanh(iInput, oOutput);;
}

double CActivationFunction::DerivativeActivationFunctions(double iInput)
{
	return DerivativeTanh(iInput);
}

bool CActivationFunction::FastSigmoid(double iInput, double &oOutput)
{
	oOutput = iInput / (1 + abs(iInput));

	return oOutput > 0 ? true : false;
}

bool CActivationFunction::ReLU(double iInput, double &oOutput)
{
	oOutput = iInput > 0 ? iInput : 0;

	return oOutput > 0 ? true : false;
}

double CActivationFunction::DerivativeReLU(double iInput)
{
	return iInput > 0 ? 1 : 0;
}

bool CActivationFunction::Tanh(double iInput, double &oOutput)
{
	oOutput = tanh(iInput);

	return oOutput > 0 ? true : false;
}

double CActivationFunction::DerivativeTanh(double iInput)
{
	double realOut = 1 - (tanh(iInput)*tanh(iInput));

	return realOut;
}
