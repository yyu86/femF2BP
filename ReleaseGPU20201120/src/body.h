//
//  Created by Yang Yu & Yunfeng Gao on 10/15/20.
//
//

#ifndef _BODY_H_
#define _BODY_H_

#include <iostream>
#include <string>
#include <fstream>
#include "mat3d.h"

using namespace std;

typedef struct fem {
    
    double TotalMass;
    Vector MassCenter,InertVec;
    Matrix Inertia;
    int VertNum, FaceNum, NodeNum, ElemNum;
    double **Vertices;
    int **Faces;
    double *NodeDensities;
    double *NodeWeights;
    double **Nodes;
    int **Elements;
    double *ElemJS;
    
} FEM;

void LoadFEM(string filename, FEM &fe);
void AllocFEM(FEM &fe);

#endif
