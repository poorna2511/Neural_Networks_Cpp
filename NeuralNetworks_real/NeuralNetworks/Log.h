#pragma once

#include <iostream>
#include <map>
#include "Windows.h"

using namespace std;

map<string, int> cmap = {
						{"Blue"    , 1 },
						{"Green"   , 2 },
						{"Red"     , 4 },
						{"Yellow"  , 6 },
						{"White"   , 7 } };

HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);

void SetConsoleTextColor(string s)
{
	SetConsoleTextAttribute(h, cmap[s]);
}

void Log(const string &iStrlog, const string &iStrColor)
{
	SetConsoleTextAttribute(h, cmap[iStrColor]);
	cout << iStrlog << endl;
}