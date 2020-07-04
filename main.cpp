#include <iostream>
#include <vector>
#include <random>
#include "vectio.h"
#include "nnmath.h"
using namespace std;


class neuralnet{
private:
  random_device rd{};
  mt19937_64 generator{rd()};
  uniform_real_distribution<float> dist;
  
public:
  vector<int> sizes = {2,3,2}; //specify network size

  //general network variables:
  vector<vector<vector<float>>> weights;
  vector<vector<float>> biases;
  vector<vector<float>> activations;
  vector<vector<float>> presigactivations;

  //variables for backprop:
  vector<vector<float>> tpw;
  vector<float> nabC;
  vector<vector<float>> sigprime;
  vector<vector<float>> dp;
  vector<float> desiredoutput;
  vector<vector<float>> delta;
  
  void feedforwards(){
  for (int layer = 0; layer < this->activations.size()-1; layer++){
    for (int neuron = 0; neuron < this->activations[layer+1].size(); neuron++){
      dot(this->weights[layer][neuron], this->activations[layer], this->activations[layer+1][neuron]);
      }
    vectadd(this->activations[layer+1], this->biases[layer], this->activations[layer+1]);
    this->presigactivations[layer+1] = this->activations[layer+1];
    vectsigmoid(activations[layer+1], this->activations[layer+1]);
    }
  }

  void backprop(){
    //calc sigprime for all non-input layers
    for (int layer = 0; layer < this->sigprime.size(); layer++){
      vectsigmoidprime(this->presigactivations[layer+1], this->sigprime[layer]);
    }
    //get error in last layer
    vectsub(this->activations.back(), this->desiredoutput, this->nabC);
    hadamard(this->nabC, this->sigprime.back(), this->delta.back());
    //propogate backwards
    for (int layer = this->delta.size()-2; layer > -1; layer--){
      transpose(this->weights[layer+1], this->tpw); //get transpose of weights[layer+1] and save in tpw
      for (int neuron = 0; neuron < this->activations[layer+1].size(); neuron++){ //get dotprod of tpw and delta[layer+1]
	this->dp[layer][neuron] = dot(this->tpw[neuron], this->delta[layer+1], this->dp[layer][neuron]);
      }
      hadamard(this->dp[layer], this->sigprime[layer], this->delta[layer]);
    }
  }
  
  //initialise variables
  neuralnet(){
    typename std::uniform_real_distribution<float>::param_type prms (float{0}, float{1});
    this->dist.param (prms);  
    //fill weights
    this->weights.resize(this->sizes.size()-1);
    for (int layer = 0; layer < this->weights.size(); layer++){
      this->weights[layer].resize(this->sizes[layer+1]);
      for (int sublayer = 0; sublayer < this->weights[layer].size(); sublayer++){
	this->weights[layer][sublayer].resize(this->sizes[layer]);
	for (int item = 0; item < this->weights[layer][sublayer].size(); item++){
	  this->weights[layer][sublayer][item] = dist(generator);
	}
      }
    }  
    //fill biases
    this->biases.resize(this->sizes.size()-1);
    for (int layer = 0; layer < this->biases.size(); layer++){
      this->biases[layer].resize(this->sizes[layer+1]);
      for (int item = 0; item < this->biases[layer].size(); item++){
	this->biases[layer][item] = dist(generator);
      }
    }
    //resize activations
    this->activations.resize(this->sizes.size());
    for (int layer = 0; layer < this->activations.size(); layer++){
      this->activations[layer].resize(this->sizes[layer]);
    }
    //resize presigactivations vector
    this->presigactivations.resize(this->sizes.size());
    for (int layer = 0; layer < this->presigactivations.size(); layer++){
      this->presigactivations[layer].resize(this->sizes[layer]);
    }

    
    //do stuff on backprop variables:
    this->nabC.resize(this->sizes.back());
    this->sigprime.resize(this->sizes.size()-1);
    this->dp.resize(this->sizes.size()-2);
    this->desiredoutput.resize(this->sizes.back());
    this->delta.resize(this->sizes.size()-1);
    for (int layer = 0; layer < this->dp.size(); layer++){
      this->dp[layer].resize(this->sizes[layer+1]);
    }
    for (int layer = 0; layer < this->delta.size(); layer++){
      this->delta[layer].resize(this->sizes[layer+1]);
    }
    for (int layer = 0; layer < this->sigprime.size(); layer++){
      this->sigprime[layer].resize(this->sizes[layer+1]);
    }


  }
};

float MSE(const vector<float>& outactivs,
	  const vector<float>& desiredout,
	  float& cost){
  cost = 0;
  for (int neuron = 0; neuron < outactivs.size(); neuron++){
    cost += pow((outactivs[neuron]-desiredout[neuron]) ,2);
  }
  return cost;
}
int main(){
  neuralnet net1;

  vector<vector<float>> nabla_b;
  float cost;
  nabla_b.resize(net1.biases.size());
  for (int layer = 0; layer < nabla_b.size(); layer++){
    nabla_b.resize(net1.biases[layer].size());
  }
  
  float eta = 1;
  
  net1.activations[0] = {0.4, 0.5};
  net1.feedforwards();
  net1.backprop();
  
  cout << "costbefore:" << endl;
  MSE(net1.activations.back(), net1.desiredoutput, cost);
  cout << cost << endl;
  
  //update biases
  for (int layer = 0; layer < net1.biases.size(); layer++){
    nabla_b[layer] = net1.delta[layer];
    scalarmultiply(nabla_b[layer], eta, nabla_b[layer]);
    vectsub(net1.biases[layer], nabla_b[layer], net1.biases[layer]);
  }

  net1.feedforwards();
  
  cout << "costafter:" << endl;
  MSE(net1.activations.back(), net1.desiredoutput, cost);
  cout << cost << endl;
  
  
}
