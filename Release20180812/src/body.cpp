#include "body.h"

void LoadFEM(string filename, FEM &fe)
{
    int i, j, a, b;
    
    const char* file;
    FILE * infile;
    
    file = filename.c_str();
    //
    infile = fopen(file, "r");
    //
    fscanf(infile, "%lf", &fe.TotalMass);
    fscanf(infile, "%lf%lf%lf", &fe.MassCenter[0],&fe.MassCenter[1],&fe.MassCenter[2]);
    for (i=0; i<3; i++) fscanf(infile, "%lf%lf%lf", &fe.Inertia[i][0],&fe.Inertia[i][1],&fe.Inertia[i][2]);
    //
    fscanf(infile, "%d%d%d%d", &fe.VertNum,&fe.FaceNum,&fe.NodeNum,&fe.ElemNum);
    //
    AllocFEM(fe);
    //
    for(i=0; i<fe.VertNum; i++) fscanf(infile, "%lf%lf%lf", &fe.Vertices[i][0],&fe.Vertices[i][1],&fe.Vertices[i][2]);
    //
    for(i=0; i<fe.FaceNum; i++) fscanf(infile, "%d%d%d", &fe.Faces[i][0],&fe.Faces[i][1],&fe.Faces[i][2]);
    //
    for(i=0; i<fe.NodeNum; i++) fscanf(infile, "%lf%lf%lf%lf%lf", &fe.NodeDensities[i],&fe.NodeWeights[i],&fe.Nodes[i][0],&fe.Nodes[i][1],&fe.Nodes[i][2]);
    //
    for(i=0; i<fe.ElemNum; i++) fscanf(infile,"%d%d%d%d%lf",&fe.Elements[i][0],&fe.Elements[i][1],&fe.Elements[i][2],&fe.Elements[i][3],&fe.ElemJS[i]);
    //
    fclose(infile);
    //
    for(i=0; i<3; i++) fe.InertVec[i] = fe.Inertia[i][i];
    //
}

void AllocFEM(FEM &fe)
{
    int i;
    
    fe.Vertices = new double*[fe.VertNum];
    for (i=0; i<fe.VertNum; i++) fe.Vertices[i] = new double[3];
    
    fe.Faces = new int*[fe.FaceNum];
    for (i=0; i<fe.FaceNum; i++) fe.Faces[i] = new int[3];
    
    fe.NodeDensities = new double[fe.NodeNum];
    fe.NodeWeights = new double[fe.NodeNum];
    
    fe.Nodes = new double*[fe.NodeNum];
    for (i=0; i<fe.NodeNum; i++) fe.Nodes[i] = new double[3];
    
    fe.Elements = new int*[fe.ElemNum];
    for (i=0; i<fe.ElemNum; i++) fe.Elements[i] = new int[4];
    
    fe.ElemJS = new double[fe.ElemNum];
    
    //cout<<"8"<<endl;
    
}
