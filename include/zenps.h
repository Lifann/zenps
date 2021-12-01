/* Copyright 2015 Lifann <xhlyfan@gmail.com>. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

namespace zenps {

ZenPS* new_zenps();

struct TableInfo {
  Dtype key_dtype;
  Dtype value_dtype;
  int id;
  int dim;
}

struct TensorInfo zenps_query_tensor(ZenPS* ps, int id);

__host__ __device__ Status zenps_get_l(ZenPS* ps, const int* keys, float* values, const size_t size);
__host__ __device__ Status zenps_get_ll(ZenPS* ps, const size_t* keys, float* values, const size_t size);

__host__ __device__ Status zenps_set_l(ZenPS* ps, const int* keys, float* values, const size_t size, const size_t embedding_size);
__host__ __device__ Status zenps_set_ll(ZenPS* ps, const int* keys, float* values, const size_t size, const size_t embedding_size);

__host__ __device__ Status zenps_size(ZenPS* ps, int64* size);

// TODO(Lifann)
// __host__ __device__ int zenps_size(ZenPS* ps);


}  // namespace zenps
