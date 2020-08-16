#include <vector>
#include "mnist.h"

using namespace std;

int main(){
  vector<int> labels = loadlabels();
  vector<vector<float>> imgs = loadimages();
  cout << labels.size() << endl;
  cout << imgs[0].size() << endl;
}
