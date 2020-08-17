#include <iostream>
#include <vector>
#include <random>
#include "vectio.h"
#include "nnmath.h"
#include "mnist.h"
using namespace std;


class neuralnet{
private:
  random_device rd{};
  mt19937_64 generator{rd()};
  uniform_real_distribution<float> dist;

public:
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
    vectsigmoid(this->activations[layer+1], this->activations[layer+1]);
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
        dot(this->tpw[neuron], this->delta[layer+1], this->dp[layer][neuron]);
      }
      hadamard(this->dp[layer], this->sigprime[layer], this->delta[layer]);
    }
  }

  //initialise variables
  neuralnet(vector<int> sizes){
    typename std::uniform_real_distribution<float>::param_type prms (float{-1}, float{1});
    this->dist.param (prms);
    //fill weights
    this->weights.resize(sizes.size()-1);
    for (int layer = 0; layer < this->weights.size(); layer++){
      this->weights[layer].resize(sizes[layer+1]);
      for (int sublayer = 0; sublayer < this->weights[layer].size(); sublayer++){
	this->weights[layer][sublayer].resize(sizes[layer]);
	for (int item = 0; item < this->weights[layer][sublayer].size(); item++){
	  this->weights[layer][sublayer][item] = dist(generator);
	}
      }
    }
    //fill biases
    this->biases.resize(sizes.size()-1);
    for (int layer = 0; layer < this->biases.size(); layer++){
      this->biases[layer].resize(sizes[layer+1]);
      for (int item = 0; item < this->biases[layer].size(); item++){
	this->biases[layer][item] = dist(generator);
      }
    }
    //resize activations
    this->activations.resize(sizes.size());
    for (int layer = 0; layer < this->activations.size(); layer++){
      this->activations[layer].resize(sizes[layer]);
    }
    //resize presigactivations vector
    this->presigactivations.resize(sizes.size());
    for (int layer = 0; layer < this->presigactivations.size(); layer++){
      this->presigactivations[layer].resize(sizes[layer]);
    }


    //do stuff on backprop variables:
    this->nabC.resize(sizes.back());
    this->sigprime.resize(sizes.size()-1);
    this->dp.resize(sizes.size()-2);
    this->desiredoutput.resize(sizes.back());
    this->delta.resize(sizes.size()-1);
    for (int layer = 0; layer < this->dp.size(); layer++){
      this->dp[layer].resize(sizes[layer+1]);
    }
    for (int layer = 0; layer < this->delta.size(); layer++){
      this->delta[layer].resize(sizes[layer+1]);
    }
    for (int layer = 0; layer < this->sigprime.size(); layer++){
      this->sigprime[layer].resize(sizes[layer+1]);
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
  neuralnet net1({784,32,10});

  float cost;

  float eta = 1;

  cout << "loading images" << endl;
  vector<vector<float>> imgs = loadimages();
  vector<int> labels = loadlabels();
  cout << "images loaded" << endl;
  cout << "normalizing data" << endl;
  for (auto& i : imgs){
    for (auto& i2 : i){
      i2 = i2/255;
    }
  }
  //create nabla_b
  vector<vector<float>> nabla_b;
  nabla_b.resize(net1.biases.size());
  for (int layer = 0; layer < net1.biases.size(); layer++){
    nabla_b.resize(net1.biases[layer].size());
  }
  //create nabla_w
  vector<vector<vector<float>>> nabla_w;
  nabla_w.resize(net1.weights.size());
  for (int layer = 0; layer < net1.weights.size(); layer++){
    nabla_w[layer].resize(net1.weights[layer].size());
    for (int sublayer = 0; sublayer < net1.weights[layer].size(); sublayer++){
      nabla_w[layer][sublayer].resize(net1.weights[layer][sublayer].size());
    }
  }
  
  for (int epoch = 0; epoch < 1; epoch++){
    for (int image = 0; image < 20000; image++){
      net1.activations[0] = imgs[image];
      net1.desiredoutput = {0,0,0,0,0,0,0,0,0,0};
      net1.desiredoutput[labels[image]] = 1;
      
      net1.feedforwards();
      net1.backprop();

      MSE(net1.activations.back(), net1.desiredoutput, cost);
      cout << cost << endl;

      //update biases
      for (int layer = 0; layer < net1.biases.size(); layer++){
	nabla_b[layer] = net1.delta[layer];
	vectbyscalarmultiply(nabla_b[layer], eta, nabla_b[layer]);
	vectsub(net1.biases[layer], nabla_b[layer], net1.biases[layer]);
      }
      //update weights
      for (int layer = 0; layer < net1.weights.size(); layer++){
	for (int neuron = 0; neuron < net1.weights[layer].size(); neuron++){
	  vectbyscalarmultiply(net1.activations[layer], net1.delta[layer][neuron], nabla_w[layer][neuron]);
	  vectbyscalarmultiply(nabla_w[layer][neuron], eta, nabla_w[layer][neuron]);
	  vectsub(net1.weights[layer][neuron], nabla_w[layer][neuron], net1.weights[layer][neuron]);
	}
      }
    }
  }
}
