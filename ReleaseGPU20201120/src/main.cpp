//
//  main.cpp
//  
//
//  Created by Yang Yu & Yunfeng Gao on 10/15/20.
//
//

#include <iostream>
#include "initialization.h"

//

using namespace std;
//
int main(int argc,char *argv[])
{
    FEM alpha, beta;
    PHASE ph0;
    ODE_OPTION odeopt;
    
    setbuf(stdout,(char *)NULL);
    
    if (argc!=2)
    {
        (void) fprintf(stderr,"Usage: %s file\n",argv[0]);
        exit(1);
    }
    
    //cout<<"1"<<endl;
    
    Initialization(argv[1], alpha, beta, ph0, odeopt);

    //cout<<"2"<<endl;
    
    Simulation(alpha, beta, ph0, odeopt);

    return 0;
    
}


