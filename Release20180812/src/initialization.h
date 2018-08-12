//
//  initialization.h
//  
//
//  Created by Yang Yu on 10/15/17.
//
//

#ifndef _INITIALIZATION_H_
#define _INITIALIZATION_H_

#include <fstream>
#include <algorithm>
#include "binary.h"

using namespace std;

void Initialization(char* parafile, FEM &alpha, FEM &beta, PHASE &ph0, ODE_OPTION &odeopt);
void LoadParas(char* parafile, string &alphafile, string &betafile, PHASE &ph0, ODE_OPTION &odeopt);

#endif /* */
