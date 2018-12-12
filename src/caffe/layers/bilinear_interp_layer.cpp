#include <vector>

//#include <boost/shared_ptr.hpp>
// #include <gflags/gflags.h>
// #include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/bilinear_interp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__attribute__ ((visibility ("hidden") ))
void BiLinearInterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	//string prefix = "\t\tBilinear Interp Layer:: LayerSetUp: \t";

	if(this->layer_param_.bilinear_interp_param().to_compute_du()) {
		to_compute_dU_ = true;
	}

	// std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

	num_sample_pts = bottom[1]->shape(2);

	// std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
__attribute__ ((visibility ("hidden") ))
void BiLinearInterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	//string prefix = "\t\tBilinear Interp Layer:: Reshape: \t";

	//if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	//bottom[0] should be input feature map
	//bottom[1] should N x 2 x num_sample_pts (x,y coordinates on bottom[0])

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape V
	vector<int> shape(3);

	
	shape[0] = N;
	shape[1] = num_sample_pts;
	shape[2] = C;

	top[0]->Reshape(shape);

	//if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

// Sampler, sample input channel 'pic' from px, py
template <typename Dtype>
__attribute__ ((visibility ("hidden") ))
Dtype BiLinearInterpLayer<Dtype>::transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py) {

	bool debug = false;

	//string prefix = "\t\tBilinear Interp Layer:: transform_forward_cpu: \t";

	//if(debug) std::cout<<prefix<<"Starting!\t"<<std::endl;
	//if(debug) std::cout<<prefix<<"(px, py) = ("<<px<<", "<<py<<")"<<std::endl;

	Dtype res = (Dtype)0.;

	Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;

	//if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;
	//if(debug) std::cout<<prefix<<"1: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		//if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x) + 1; n = floor(y); w = 0;
	//if(debug) std::cout<<prefix<<"2: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		//if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x); n = floor(y) + 1; w = 0;
	//if(debug) std::cout<<prefix<<"3: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		//if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	//if(debug) std::cout<<prefix<<"4: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		//if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	//if(debug) std::cout<<prefix<<"Finished. \tres = "<<res<<std::endl;

	return res;
}

template <typename Dtype>
__attribute__ ((visibility ("hidden") ))
void BiLinearInterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	//string prefix = "\t\tBilinear Interp Layer:: Forward_cpu: \t";

	// CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
			// " CHECK in st_layer.cpp file. Line number: 240-241." << std::endl;

	//if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	const Dtype* U = bottom[0]->cpu_data();
	const Dtype* sample_pts = bottom[1]->cpu_data();

	// Sampling
	Dtype* V = top[0]->mutable_cpu_data();
	caffe_set(top[0]->count(), (Dtype)0, V);

	// for each input
	for(int i = 0; i < N; ++i) {
		const Dtype* coordinates = sample_pts + (num_sample_pts * 2) * i;
		Dtype px, py;
		// Do sampling
		for(int s = 0; s < num_sample_pts; ++s) {
			for(int j = 0; j < C; ++j) {

				px = coordinates[s];
				py = coordinates[s + num_sample_pts];

				V[top[0]->offset(i, s, j)] = transform_forward_cpu(
						U + bottom[0]->offset(i, j, 0, 0), px, py);
			}
		}	
	}

	//if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
__attribute__ ((visibility ("hidden") ))
void BiLinearInterpLayer<Dtype>::transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
		const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy) {

	bool debug = false;

	//string prefix = "\t\tBilinear Interp Layer:: transform_backward_cpu: \t";

	//if(debug) std::cout<<prefix<<"Starting!"<<std::endl;

	Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;
	//if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;
	//if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x) + 1; n = floor(y); w = 0;
	//if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x); n = floor(y) + 1; w = 0;
	//if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	//if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				//if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				//if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	//if(debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
__attribute__ ((visibility ("hidden") ))
void BiLinearInterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		//string prefix = "\t\tBilinear Interp Layer:: Backward_cpu: \t";

		// CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
		// 		" CHECK in st_layer.cpp file. Line number: 420-421." << std::endl;

		//if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

		const Dtype* dV = top[0]->cpu_diff();
		const Dtype* U = bottom[0]->cpu_data();
		const Dtype* sample_pts = bottom[1]->cpu_data();


		Dtype* dU = bottom[0]->mutable_cpu_diff();
		Dtype* sample_pts_diff = bottom[1]->mutable_cpu_diff();

		caffe_set(bottom[0]->count(), (Dtype)0, dU);
		caffe_set(bottom[1]->count(), (Dtype)0, sample_pts_diff);

		for(int i = 0; i < N; ++i) {

			const Dtype* coordinates = sample_pts + (num_sample_pts * 2) * i;
			Dtype* coordinates_diff = sample_pts_diff + (num_sample_pts * 2) * i;

			Dtype px, py, delta_dpx, delta_dpy;

			for(int s = 0; s < num_sample_pts; ++s) {


				px = coordinates[s];
				py = coordinates[s + num_sample_pts];

				for(int j = 0; j < C; ++j) {

					delta_dpx = delta_dpy = (Dtype)0.;

					transform_backward_cpu(dV[top[0]->offset(i, s, j)], U + bottom[0]->offset(i, j, 0, 0),
							px, py, dU + bottom[0]->offset(i, j, 0, 0), delta_dpx, delta_dpy);

					coordinates_diff[s] += delta_dpx;
					coordinates_diff[s + num_sample_pts] += delta_dpy;
				}
			}
		}

		//if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(BiLinearInterpLayer);
#endif

INSTANTIATE_CLASS(BiLinearInterpLayer);
REGISTER_LAYER_CLASS(BiLinearInterp);

}  // namespace caffe
