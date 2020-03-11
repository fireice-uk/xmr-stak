/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <chrono>
#include <thread>
#include "xmrstak/backend/nvidia/nvcc_code/cryptonight.hpp"
#include "../nvcc_code/cuda_device.hpp"

namespace RandomX_Coinevo {
    #include "configuration.h"
    #define fillAes4Rx4 fillAes4Rx4_v104
    #include "../RandomX/common.hpp"
}
