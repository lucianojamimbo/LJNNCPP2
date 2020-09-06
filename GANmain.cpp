#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
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
  float cost;

  //variables for backprop:
  vector<vector<float>> tpw;
  vector<float> nabC;
  vector<vector<float>> sigprime;
  vector<vector<float>> dp;
  vector<float> desiredoutput;
  vector<vector<float>> delta;
  vector<vector<float>> nabla_b;
  vector<vector<vector<float>>> nabla_w;
  vector<vector<vector<float>>> nabla_w_temp;

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
    
    //calc nabla c
    CEderivative(this->activations.back(), this->desiredoutput, this->nabC); //cross-entropy derivative
    //MSEderivative(this->activations.back(), this->desiredoutput, this->nabC); //mean squared error derivative
    
    //calc delta.back()
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

  void updateparams(const float& eta,
		    const int& batch_size,
		    const float& lambda,
		    const int& datasize){
    //update biases
    for (int layer = 0; layer < this->biases.size(); layer++){
      vectbyscalarmultiply(this->nabla_b[layer], eta/batch_size, this->nabla_b[layer]);
      vectsub(this->biases[layer], this->nabla_b[layer], this->biases[layer]);
    }
    //apply regularisation to weights:
    for (int layer = 0; layer < this->weights.size(); layer++){
      for (int neuron = 0; neuron < this->weights[layer].size(); neuron++){
	vectbyscalarmultiply(this->weights[layer][neuron], 1-((eta*lambda)/datasize), this->weights[layer][neuron]);
      }
    }
    //update weights
    for (int layer = 0; layer < this->weights.size(); layer++){
      for (int neuron = 0; neuron < this->weights[layer].size(); neuron++){
	vectbyscalarmultiply(this->nabla_w[layer][neuron], eta/batch_size, this->nabla_w[layer][neuron]);
	vectsub(this->weights[layer][neuron], this->nabla_w[layer][neuron], this->weights[layer][neuron]);
      }
    }

  }

  //initialise variables
  neuralnet(const vector<int>& sizes){

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

    //resize nabla_b
    this->nabla_b.resize(this->biases.size());
    for (int layer = 0; layer < this->biases.size(); layer++){
      this->nabla_b[layer].resize(this->biases[layer].size());
      for (int item = 0; item < this->nabla_b[layer].size(); item++){
	this->nabla_b[layer][item] = 1;
      }
    }

    //resize nabla_w
    this->nabla_w.resize(this->weights.size());
    for (int layer = 0; layer < this->weights.size(); layer++){
      this->nabla_w[layer].resize(this->weights[layer].size());
      for (int sublayer = 0; sublayer < this->weights[layer].size(); sublayer++){
	this->nabla_w[layer][sublayer].resize(this->weights[layer][sublayer].size());
      }
    }

    //resize nabla_w_temp
    this->nabla_w_temp.resize(this->weights.size());
    for (int layer = 0; layer < this->weights.size(); layer++){
      this->nabla_w_temp[layer].resize(this->weights[layer].size());
      for (int sublayer = 0; sublayer < this->weights[layer].size(); sublayer++){
	this->nabla_w_temp[layer][sublayer].resize(this->weights[layer][sublayer].size());
      }
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

class gennet : public neuralnet{
public:
  gennet(const vector<int>& sizes) : neuralnet(sizes){
    // generator-specific construction code
    typename std::uniform_real_distribution<float>::param_type prms (float{0}, float{1});
    dist.param (prms);
  }
  
  void generate(){
    for (int neuron = 0; neuron < this->activations[0].size(); neuron++){
      this->activations[0][neuron] = dist(generator);
    }
    this->feedforwards();
  }
  
private:
  random_device rd{};
  mt19937_64 generator{rd()};
  uniform_real_distribution<float> dist;
};

class discnet : public neuralnet{
  public:
    discnet(const vector<int>& sizes) : neuralnet(sizes){
        // Add any discriminator-specific construction code here, if necessary
    }

  void SGDstep(const int& minibatchsize,
	       const float& eta,
	       const float& lambda,
	       const int& datasize,
	       const vector<vector<float>>& minidatabatch,
	       const vector<float>& minidatabatchlabels,
	       const vector<int>& shuffledata){
    
      for (int batchiter = 0; batchiter < minibatchsize; batchiter++){
        this->activations[0] = minidatabatch[shuffledata[batchiter]];
        this->desiredoutput[0] = minidatabatchlabels[shuffledata[batchiter]];
        this->feedforwards();
        this->backprop();

	
	//calc nabla_b
	for (int layer = 0; layer < this->biases.size(); layer++){
	  vectadd(this->nabla_b[layer], this->delta[layer], this->nabla_b[layer]);
	}
	//calc nabla_w
	for (int layer = 0; layer < this->weights.size(); layer++){
	  for (int neuron = 0; neuron < this->weights[layer].size(); neuron++){
	    vectbyscalarmultiply(this->activations[layer], this->delta[layer][neuron], this->nabla_w_temp[layer][neuron]);
	    vectadd(this->nabla_w[layer][neuron], this->nabla_w_temp[layer][neuron], this->nabla_w[layer][neuron]);
	  }
	}
      }
      //update network parameters based on n_w and n_b:
      this->updateparams(eta, minibatchsize, lambda, datasize);
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

float CE(const vector<float>& outactivs,
	 const vector<float>& desiredout,
	 float& cost){
  cost = 0;
  for (int neuron = 0; neuron < outactivs.size(); neuron++){
    cost += (desiredout[neuron] * log(outactivs[neuron])) + (1-desiredout[neuron]*log(1-outactivs[neuron])); 
  }
  return cost;
}

int main(){
  //load dataset:
  cout << "Loading data" << endl;
  vector<vector<float>> imgs = loadimages();
  vector<int> labels = loadlabels();
  vector<vector<float>> testimgs = loadtestimages();
  vector<int> testlabels = loadtestlabels();
  cout << "Data loaded" << endl;
  cout << "Normalising data" << endl;
  for (auto& i : imgs){
    for (auto& i2 : i){
      i2 = i2/255;
    }
  }
  for (auto& i : testimgs){
    for (auto& i2 : i){
      i2 = i2/255;
    }
  }
  cout << "Data normalised" << endl;


  random_device rd{};
  mt19937_64 generator{rd()};
  uniform_real_distribution<float> dist;
  typename std::uniform_real_distribution<float>::param_type prms (float{0}, float{1});
  dist.param (prms);
  
  gennet gnet({10,32,784}); //generator net
  discnet dnet({784,32,1}); //discriminator net


  int minibatchsize = 20;
  int globalimgcount = 0;
  float eta = 1;
  float lambda = 1;
  int datasize = 60000;
  
  vector<vector<float>> minidatabatch;
  minidatabatch.resize(minibatchsize);
  for (auto& i : minidatabatch){
    i.resize(784);
  }
  vector<float> minidatabatchlabels;
  minidatabatchlabels.resize(minibatchsize);
  
  //create a vector with a length the same as the size of our data,
  //then make shuffledata[i] = i.
  //then we can shuffle shuffledata every epoch and use it to index our data in a random way without shuffling our data directly.
  //which is probably faster since less memory is moved around.
  auto rng = std::default_random_engine {};
  vector<int> shuffledata;
  shuffledata.resize(minibatchsize);
  for (int i = 0; i < minibatchsize; i++){
    shuffledata[i] = i;
  }
  
  for (int batch = 0; batch < (imgs.size()*2)/minibatchsize; batch++){
    shuffle(begin(shuffledata), end(shuffledata), rng);
    //create a batch of data:
    for (int img = 0; img < minibatchsize/2; img++){
      minidatabatch[img] = imgs[globalimgcount];
      minidatabatchlabels[img] = 0; //desired output for real instance = 0 (meaning 0% probability image is fake)
      globalimgcount += 1;
      gnet.generate();
      minidatabatch[img+(minibatchsize/2)] = gnet.activations.back();
      minidatabatchlabels[img+(minibatchsize/2)] = 1; //desired output for fake instance = 1 (meaning 100% probability image is fake)
    }
    dnet.SGDstep(minibatchsize, eta, lambda, datasize, minidatabatch, minidatabatchlabels, shuffledata);
  }
}
