#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  relu_channel_slope_.clear();
    
  const int channels=bottom[0]->channels();
  Dtype negative_slope=this->layer_param_.relu_param().negative_slope();

  for (int i=0;i<channels;i++) {
      relu_channel_slope_.push_back(negative_slope);
  }


}



template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int dim= bottom[0]->count()/num;
  const int spatial_dim=bottom[0]->height()*bottom[0]->width();
  for (int i = 0; i < num; ++i) {
      for (int j=0; j<channels; ++j) {
          Dtype negative_slope=relu_channel_slope_[j];
          for (int k=0; k<spatial_dim; ++k) {
              int ind=i*dim+j*spatial_dim+k;
                top_data[ind] = std::max(bottom_data[ind], Dtype(0))
        + negative_slope * std::min(bottom_data[ind], Dtype(0));
          }
      }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int num = (*bottom)[0]->num();
    const int channels = (*bottom)[0]->channels();
    const int dim= (*bottom)[0]->count()/num;
    const int spatial_dim=(*bottom)[0]->height() * (*bottom)[0]->width();
    for (int i = 0; i < num; ++i) {
        for (int j=0; j<channels; ++j) {
            Dtype negative_slope=relu_channel_slope_[j];
                for (int k=0; k<spatial_dim; ++k) {
                    int ind=i*dim+j*spatial_dim+k;
                    bottom_diff[ind] = top_diff[ind] * ((bottom_data[ind] > 0)
                        + negative_slope * (bottom_data[ind] <= 0));
                }

        }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
