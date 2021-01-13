//
//  initialization.cpp
//  
//
//  Created by Yang Yu & Yunfeng Gao on 10/15/20.
//
//

#include "initialization.h"

void Initialization(char* parafile, FEM &alpha, FEM &beta, PHASE &ph0, ODE_OPTION &odeopt)
{
    string alphafile, betafile;
    
    //cout<<"3"<<endl;
    LoadParas(parafile, alphafile, betafile, ph0, odeopt);

    //cout<<"4"<<endl;
    LoadFEM(alphafile, alpha);
    
    //cout<<"5"<<endl;
    LoadFEM(betafile, beta);

    //Note: we do not check the alignment of the body-fixed frames, which MUST be defined as stated in PAR file,
    //      otherwise the simulation results would be wrong!
}

//

void LoadParas(char* parafile, string &alphafile, string &betafile, PHASE &ph0, ODE_OPTION &odeopt)
{
    int i, numchar, id, n;
    int flag;
    double AlphaEuler[3], BetaEuler[3];

    ifstream infile;
    string line, core, name, value;
    
    flag = 0;

    infile.open(parafile);
    
    while(getline(infile,line))
    {
        numchar = line.length();
        for (n=0; n<numchar; n++) if (line[n] == '#') break;
        core.assign(line,0,n);
        //
        core.erase(std::remove(core.begin(), core.end(), '\t'), core.end());
        core.erase(std::remove(core.begin(), core.end(), ' '), core.end());
        //
        if (! core.empty())
        {
            //
            numchar = core.length();
            id = core.find('=');
            assert((id>0) && (id+1<numchar));
            name.assign(core,0,id);
            n = numchar - id - 1;
            assert(n>0);
            value.assign(core,id+1,n);
            //
            if (name == "AlphaFile")            {alphafile = value;                              flag++;}
            else if (name == "BetaFile")        {betafile = value;                              flag++;}
            //
            else if (name == "AlphaPosX")       {ph0.AlphaPos[0] = atof(value.c_str());         flag++;}
            else if (name == "AlphaPosY")       {ph0.AlphaPos[1] = atof(value.c_str());         flag++;}
            else if (name == "AlphaPosZ")       {ph0.AlphaPos[2] = atof(value.c_str());         flag++;}
            else if (name == "AlphaVelX")       {ph0.AlphaVel[0] = atof(value.c_str());         flag++;}
            else if (name == "AlphaVelY")       {ph0.AlphaVel[1] = atof(value.c_str());         flag++;}
            else if (name == "AlphaVelZ")       {ph0.AlphaVel[2] = atof(value.c_str());         flag++;}
            else if (name == "AlphaPhi")        {AlphaEuler[0] = atof(value.c_str());           flag++;}
            else if (name == "AlphaTheta")      {AlphaEuler[1] = atof(value.c_str());           flag++;}
            else if (name == "AlphaVarphi")     {AlphaEuler[2] = atof(value.c_str());           flag++;}
            //
            else if (name == "AlphaAngVelX")    {ph0.AlphaAngVel[0] = atof(value.c_str());      flag++;}
            else if (name == "AlphaAngVelY")    {ph0.AlphaAngVel[1] = atof(value.c_str());      flag++;}
            else if (name == "AlphaAngVelZ")    {ph0.AlphaAngVel[2] = atof(value.c_str());      flag++;}
            //
            else if (name == "BetaPosX")        {ph0.BetaPos[0] = atof(value.c_str());          flag++;}
            else if (name == "BetaPosY")        {ph0.BetaPos[1] = atof(value.c_str());          flag++;}
            else if (name == "BetaPosZ")        {ph0.BetaPos[2] = atof(value.c_str());          flag++;}
            else if (name == "BetaVelX")        {ph0.BetaVel[0] = atof(value.c_str());          flag++;}
            else if (name == "BetaVelY")        {ph0.BetaVel[1] = atof(value.c_str());          flag++;}
            else if (name == "BetaVelZ")        {ph0.BetaVel[2] = atof(value.c_str());          flag++;}
            else if (name == "BetaPhi")         {BetaEuler[0] = atof(value.c_str());            flag++;}
            else if (name == "BetaTheta")       {BetaEuler[1] = atof(value.c_str());            flag++;}
            else if (name == "BetaVarphi")      {BetaEuler[2] = atof(value.c_str());            flag++;}
            //
            else if (name == "BetaAngVelX")     {ph0.BetaAngVel[0] = atof(value.c_str());       flag++;}
            else if (name == "BetaAngVelY")     {ph0.BetaAngVel[1] = atof(value.c_str());       flag++;}
            else if (name == "BetaAngVelZ")     {ph0.BetaAngVel[2] = atof(value.c_str());       flag++;}
            //
            else if (name == "AbsTol")          {odeopt.AbsTol = atof(value.c_str());           flag++;}
            else if (name == "RelTol")          {odeopt.RelTol = atof(value.c_str());           flag++;}
            else if (name == "TimeEnd")         {odeopt.TimeEnd = atof(value.c_str());          flag++;}
            //
            else {cout<< "Unrecoganized item "<<name<<"!"<<endl; exit(1);}
            //
        }
    }
    infile.close();
    
    if (flag != NumParameter)
    {
        cout<<flag<<endl;
        cout<<"Important parameter setting missing!"<<endl;
        exit(1);
    }
    
    Euler2Quat(AlphaEuler, ph0.AlphaOrien);
    Euler2Quat(BetaEuler, ph0.BetaOrien);

}


