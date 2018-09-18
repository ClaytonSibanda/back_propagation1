#include <iostream>
#include "Neuron.h"
#include <cmath>
using namespace std;
double transferFunction(double x);

vector<vector<Neuron> > layers;
void forward_pass(vector<vector<Neuron> > &layers);
double getError(double expected,double target);

double transferFunctionDerivative(double x);
void back_pass(vector<vector<Neuron> > &layers);
void print_results(vector<vector<Neuron> > &layers);

int main() {

    vector<double> inputs ={0,1};
    vector<double> i_weights={1,1};
    Neuron i1(i_weights);
    Neuron i2(i_weights);
    i1.setInput(0);
    i2.setInput(1);
    i1.setOutput(0);
    i2.setOutput(1);

    vector<double> weights1 ={-1,0};
    vector<double> weights2 = {0,1};
    Neuron h1(weights1);
    Neuron h2(weights2);

    vector<double> weights_o1 ={1,0};
    vector<double> weights_o2 = {-1,1};
    Neuron o1(weights_o1);
    Neuron o2(weights_o2);

    layers ={{i1,i2},{h1,h2},{o1,o2}};
    //in
    forward_pass(layers);

    layers[2][0].setError(getError(layers[2][0].getOutput(),1));
    layers[2][1].setError(getError(layers[2][1].getOutput(),0));



    back_pass(layers);

    print_results(layers);
    return 0;
}

void print_results(vector<vector<Neuron> > &layers){
    cout<<"Output node outputs:\n";
    cout<<layers[2][0].getOutput()<<endl;
    cout<<layers[2][1].getOutput()<<endl;

    cout<<"\nHidden node outputs:\n";
    cout<<layers[1][0].getOutput()<<endl;
    cout<<layers[1][1].getOutput()<<endl;

    cout<<"\nOutput node errors:\n";
    cout<<layers[2][0].getError()<<endl;
    cout<<layers[2][1].getError()<<endl;

    cout<<"\nHidden node errors:\n";
    cout<<layers[1][0].getError()<<endl;
    cout<<layers[1][1].getError()<<endl;

    cout<<"\nWeights after backpass(v's)\n";

    cout<<"v11: "<<layers[1][0].getWeights()[0]<<endl;
    cout<<"v21: "<<layers[1][0].getWeights()[1]<<endl;
    cout<<"v12: "<<layers[1][1].getWeights()[0]<<endl;
    cout<<"v22: "<<layers[1][1].getWeights()[1]<<endl;

    cout<<"\nWeights after backpass(w's)\n";
    cout<<"w11: "<<layers[2][0].getWeights()[0]<<endl;
    cout<<"w21: "<<layers[2][0].getWeights()[1]<<endl;
    cout<<"w12: "<<layers[2][1].getWeights()[0]<<endl;
    cout<<"w22: "<<layers[2][1].getWeights()[1]<<endl;


}



void forward_pass(vector<vector<Neuron> > &layers){
    vector<Neuron> prev_layer;
    for(unsigned int i=1;i<layers.size();++i){
        prev_layer=layers[i-1];
        for(unsigned int j=0;j<layers[i].size();++j){
            double input= prev_layer[0].output*layers[i][j].weights[0]+prev_layer[1].output*layers[i][j].weights[1];
           // cout<<"out"<<i<<" "<<j<<": "<<transferFunction(input)<<endl;
            vector<double> inputs={prev_layer[0].output,prev_layer[1].output};
            layers[i][j].inputs=inputs;
            layers[i][j].setInput(input);
            layers[i][j].setOutput(transferFunction(input));
        }
    }

}

void back_pass(vector<vector<Neuron> > &layers){

    double h1_error= layers[1][0].getOutput()*(1-layers[1][0].getOutput())*(layers[2][0].weights[0]*layers[2][0].getError()+layers[2][1].weights[0]*layers[2][1].getError());

    double h2_error= layers[1][1].getOutput()*(1-layers[1][1].getOutput())*(layers[2][0].weights[1]*layers[2][0].getError()+layers[2][1].weights[1]*layers[2][1].getError());
    layers[1][0].setError(h1_error);
    layers[1][1].setError(h2_error);
    //cout<<"h1 error "<<h1_error<<endl;
  //  cout<<"h1 error "<<h2_error<<endl;



    double v11=layers[1][0].weights[0]+(0.1*layers[1][0].getError()*layers[1][0].inputs[0]);
    double v12=layers[1][0].weights[1]+(0.1*layers[1][0].getError()*layers[1][0].inputs[1]);
    vector<double> weights_v ={v11,v12};
    layers[1][0].setWeights(weights_v);

    double v21=layers[1][1].weights[0]+(0.1*layers[1][1].getError()*layers[1][1].inputs[0]);
    double v22=layers[1][1].weights[1]+(0.1*layers[1][1].getError()*layers[1][1].inputs[1]);
    vector<double> weights_v1 ={v21,v22};
    layers[1][1].setWeights(weights_v1);

//    cout<<"v11: "<<v11<<endl;
//cout<<"v11: "<<v12<<endl;
//cout<<"v11: "<<v21<<endl;
//cout<<"v11: "<<v22<<endl;


    double w11=layers[2][0].weights[0]+(0.1*layers[2][0].getError()*layers[2][0].inputs[0]);
  // cout<<"weight w11: "<<w11<<" weight1 "<<layers[2][0].weights[0]<<endl;

    double w12=layers[2][0].weights[1]+(0.1*layers[2][0].getError()*layers[2][0].inputs[1]);
   // cout<<"weight w12: "<<w12<<" weight1 "<<layers[2][0].weights[1]<<endl;
    vector<double> weights ={w11,w12};
    layers[2][0].setWeights(weights);
    double w21=layers[2][1].weights[0]+(0.1*layers[2][1].getError()*layers[2][1].inputs[0]);
   // cout<<"weight w21: "<<w21<<endl;
    double w22=layers[2][1].weights[1]+(0.1*layers[2][1].getError()*layers[2][1].inputs[1]);
    //cout<<"weight w22: "<<w22<<endl;
    vector<double> weights1 ={w21,w22};
    layers[2][1].setWeights(weights1);

    }



double getError(double output,double target){

    return output*(1-output)*(target-output);
}

double transferFunction(double x) {

    return 1/(1+exp(-x));
}


