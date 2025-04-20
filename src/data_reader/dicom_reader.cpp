#include "dicom_reader.h"

#include <filesystem>
#include <string>
#include <algorithm>
#include <cmath>
#include <torch/types.h>
#include <vega/dicom/file.h>
#include <vega/dictionary/dictionary.h>
#include <torch/torch.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>


namespace fs = std::filesystem;

DicomReader::DicomReader(const std::string& dictionary) {
    if (dictionary == "")
        vega::dictionary::set_dictionary(DEFAULT_DICOM_DICTIONARY);
    else
        vega::dictionary::set_dictionary(dictionary);
}

static bool zAxisComparison(vega::dicom::File &f1, vega::dicom::File &f2) {
    auto f1_pos = getDataManipulator(f1, vega::dictionary::ImagePositionPatient);
    auto f2_pos = getDataManipulator(f2, vega::dictionary::ImagePositionPatient);
    return static_cast<float>(f1_pos->begin()[2]) < static_cast<float>(f2_pos->begin()[2]);
}

std::pair<Geometry, torch::Tensor> DicomReader::readDirectory(const std::string& path) {
    for (const auto &entry: fs::directory_iterator(path)) {
        dicomFiles.emplace_back(entry.path().string());
    }
    std::sort(dicomFiles.begin(), dicomFiles.end(), zAxisComparison);

    auto bits = *getDataManipulator(dicomFiles[0], vega::dictionary::BitsAllocated)->begin();
    auto width = *getDataManipulator(dicomFiles[0], vega::dictionary::Columns)->begin();
    auto height = *getDataManipulator(dicomFiles[0], vega::dictionary::Rows)->begin();
    const uint32_t length = width * height * dicomFiles.size();
    float* volume = new float[length];
    torch::Tensor volume_tensor;
    if (bits == 16) {
        size_t index = 0;
        for (const auto& image : dicomFiles) {
            auto pixel_data = getDataManipulator(image, vega::dictionary::PixelData_OW);
            auto slope = static_cast<float>(*getDataManipulator(image, vega::dictionary::RescaleSlope)->begin());
            auto intercept = static_cast<float>(*getDataManipulator(image, vega::dictionary::RescaleIntercept)->begin());

            for (auto it = pixel_data->begin(); it != pixel_data->end(); ++it) {
                volume[index++] = it->u * slope + intercept;
            }
        }

        volume_tensor = torch::from_blob(volume, {static_cast<long>(length)}, torch::kFloat);
        volume_tensor = (volume_tensor - volume_tensor.min()) / (volume_tensor.max() - volume_tensor.min());
    }else{
        std::cerr << "Unsupported Bits Allocated: " << bits << std::endl;
        delete[] volume;
        return {};
    }

    Geometry geom;
    geom.nVoxelX = width;
    geom.nVoxelY = height;
    geom.nVoxelZ = dicomFiles.size();

    auto spacing = getDataManipulator(dicomFiles[0], vega::dictionary::PixelSpacing);
    geom.sVoxelX = static_cast<float>(spacing->begin()[0]);
    geom.sVoxelY = static_cast<float>(spacing->begin()[1]);
    geom.sVoxelZ = static_cast<float>(*getDataManipulator(dicomFiles[0], vega::dictionary::SliceThickness)->begin());

    geom.offOrigX = new float(static_cast<float>(getDataManipulator(dicomFiles[0], vega::dictionary::ImagePositionPatient)->begin()[0]));
    geom.offOrigY = new float(static_cast<float>(getDataManipulator(dicomFiles[0], vega::dictionary::ImagePositionPatient)->begin()[1]));
    geom.offOrigZ = new float(static_cast<float>(getDataManipulator(dicomFiles[0], vega::dictionary::ImagePositionPatient)->begin()[2]));

    geom.nDetecU = geom.nVoxelX;
    geom.nDetecV = geom.nVoxelY;
    geom.sDetecU = geom.sVoxelX;
    geom.sDetecV = geom.sVoxelY;
    geom.dDetecU = 1.0f;
    geom.dDetecV = 1.0f;
    geom.offDetecU = new float(0.0f);
    geom.offDetecV = new float(0.0f);

    geom.DSO = new float(1000.0f);
    geom.DSD = new float(1500.0f);

    geom.dRoll = new float(0.0f);
    geom.dPitch = new float(0.0f);
    geom.dYaw = new float(0.0f);

    geom.COR = new float(0.0f);
    geom.unitX = geom.sVoxelX;
    geom.unitY = geom.sVoxelY;
    geom.unitZ = geom.sVoxelZ;

    geom.maxLength = std::sqrt(
        (geom.nVoxelX * geom.sVoxelX) * (geom.nVoxelX * geom.sVoxelX) +
        (geom.nVoxelY * geom.sVoxelY) * (geom.nVoxelY * geom.sVoxelY) +
        (geom.nVoxelZ * geom.sVoxelZ) * (geom.nVoxelZ * geom.sVoxelZ)
    );

    geom.accuracy = 0.01f;

    auto orient = getDataManipulator(dicomFiles[0], vega::dictionary::ImageOrientationPatient);
    
    // std::cout << "Orientation: ";
    // for (size_t i = 0; i < orient->size(); ++i) {
    //     std::cout << orient->begin()[i] << " ";
    // }
    std::cout << std::endl;

    float row_x = orient->begin()[0];
    float row_y = orient->begin()[1];
    float row_z = orient->begin()[2];
    float col_x = orient->begin()[3];
    float col_y = orient->begin()[4];
    float col_z = orient->begin()[5];


    float norm_x = row_y * col_z - row_z * col_y;
    float norm_y = row_z * col_x - row_x * col_z;
    float norm_z = row_x * col_y - row_y * col_x;

    
    geom.alpha = col_x ;                    
    geom.theta = col_y;
    geom.psi   = col_z;

    // ---------------- Print ----------------
    std::cout << "Geometry:" << std::endl;
    std::cout << "  nVoxelX: " << geom.nVoxelX << std::endl;
    std::cout << "  nVoxelY: " << geom.nVoxelY << std::endl;
    std::cout << "  nVoxelZ: " << geom.nVoxelZ << std::endl;
    std::cout << "  sVoxelX: " << geom.sVoxelX << std::endl;
    std::cout << "  sVoxelY: " << geom.sVoxelY << std::endl;
    std::cout << "  sVoxelZ: " << geom.sVoxelZ << std::endl;
    std::cout << "  offOrigX: " << *geom.offOrigX << std::endl;
    std::cout << "  offOrigY: " << *geom.offOrigY << std::endl;
    std::cout << "  offOrigZ: " << *geom.offOrigZ << std::endl;
    std::cout << "  nDetecU: " << geom.nDetecU << std::endl;
    std::cout << "  nDetecV: " << geom.nDetecV << std::endl;
    std::cout << "  sDetecU: " << geom.sDetecU << std::endl;
    std::cout << "  sDetecV: " << geom.sDetecV << std::endl;
    std::cout << "  dDetecU: " << geom.dDetecU << std::endl;
    std::cout << "  dDetecV: " << geom.dDetecV << std::endl;
    std::cout << "  offDetecU: " << *geom.offDetecU << std::endl;
    std::cout << "  offDetecV: " << *geom.offDetecV << std::endl;
    std::cout << "  DSO: " << *geom.DSO << std::endl;
    std::cout << "  DSD: " << *geom.DSD << std::endl;
    std::cout << "  dRoll: " << *geom.dRoll << std::endl;
    std::cout << "  dPitch: " << *geom.dPitch << std::endl;
    std::cout << "  dYaw: " << *geom.dYaw << std::endl;
    std::cout << "  alpha: " << geom.alpha << std::endl;
    std::cout << "  theta: " << geom.theta << std::endl;
    std::cout << "  psi: " << geom.psi << std::endl;
    std::cout << "  COR: " << *geom.COR << std::endl;
    std::cout << "  maxLength: " << geom.maxLength << std::endl;
    std::cout << "  accuracy: " << geom.accuracy << std::endl;
    std::cout << "  unitX: " << geom.unitX << std::endl;
    std::cout << "  unitY: " << geom.unitY << std::endl;
    std::cout << "  unitZ: " << geom.unitZ << std::endl;

    delete[] volume;
    return {geom, volume_tensor};
}


int siddon_ray_projection(float* img, Geometry geo, float** result,float const * const angles,int nangles, const GpuIds& gpuids){
    // Prepare for MultiGPU
    int deviceCount = gpuids.GetLength();
    cudaCheckErrors("Device query fail");
    if (deviceCount == 0) {
        mexErrMsgIdAndTxt("Ax:Siddon_projection:GPUselect","There are no available device(s) that support CUDA\n");
    }
    //
    // CODE assumes
    // 1.-All available devices are usable by this code
    // 2.-All available devices are equal, they are the same machine (warning thrown)
    // Check the available devices, and if they are the same
    if (!gpuids.AreEqualDevices()) {
        mexWarnMsgIdAndTxt("Ax:Siddon_projection:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
    }
    int dev;
    
    // Check free memory
    size_t mem_GPU_global;
    checkFreeMemory(gpuids, &mem_GPU_global);

    size_t mem_image=                 (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY*(unsigned long long)geo.nVoxelZ*sizeof(float);
    size_t mem_proj=                  (unsigned long long)geo.nDetecU*(unsigned long long)geo.nDetecV*sizeof(float);
    
    // Does everything fit in the GPUs?
    const bool fits_in_memory = mem_image+2*PROJ_PER_BLOCK*mem_proj<mem_GPU_global;
    unsigned int splits=1;
    if (!fits_in_memory) {
        // Nope nope.
        // approx free memory we have. We already have left some extra 5% free for internal stuff
        // we need a second projection memory to combine multi-GPU stuff.
        size_t mem_free=mem_GPU_global-4*PROJ_PER_BLOCK*mem_proj;
        splits=mem_image/mem_free+1;// Ceil of the truncation
    }
    Geometry* geoArray = (Geometry*)malloc(splits*sizeof(Geometry));
    splitImage(splits,geo,geoArray,nangles);
    
    // Allocate axuiliary memory for projections on the GPU to accumulate partial results
    float ** dProjection_accum;
    size_t num_bytes_proj = PROJ_PER_BLOCK*geo.nDetecU*geo.nDetecV * sizeof(float);
    if (!fits_in_memory){
        dProjection_accum=(float**)malloc(2*deviceCount*sizeof(float*));
        for (dev = 0; dev < deviceCount; dev++) {
            cudaSetDevice(gpuids[dev]);
            for (int i = 0; i < 2; ++i){
                cudaMalloc((void**)&dProjection_accum[dev*2+i], num_bytes_proj);
                cudaMemset(dProjection_accum[dev*2+i],0,num_bytes_proj);
                cudaCheckErrors("cudaMallocauxiliarty projections fail");
            }
        }
    }
    
    // This is happening regarthless if the image fits on memory
    float** dProjection=(float**)malloc(2*deviceCount*sizeof(float*));
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(gpuids[dev]);
        
        for (int i = 0; i < 2; ++i){
            cudaMalloc((void**)&dProjection[dev*2+i],   num_bytes_proj);
            cudaMemset(dProjection[dev*2+i]  ,0,num_bytes_proj);
            cudaCheckErrors("cudaMalloc projections fail");
        }
    }
    
    
    //Pagelock memory for synchronous copy.
    // Lets try to make the host memory pinned:
    // We laredy queried the GPU and assuemd they are the same, thus should have the same attributes.
    int isHostRegisterSupported = 0;
#if CUDART_VERSION >= 9020
    cudaDeviceGetAttribute(&isHostRegisterSupported,cudaDevAttrHostRegisterSupported,gpuids[0]);
#endif
    // empirical testing shows that when the image split is smaller than 1 (also implies the image is not very big), the time to
    // pin the memory is greater than the lost time in Synchronously launching the memcpys. This is only worth it when the image is too big.
#ifndef NO_PINNED_MEMORY
    if (isHostRegisterSupported & (splits>1 |deviceCount>1)){
        cudaHostRegister(img, (size_t)geo.nVoxelX*(size_t)geo.nVoxelY*(size_t)geo.nVoxelZ*(size_t)sizeof(float),cudaHostRegisterPortable);
    }
#endif
    cudaCheckErrors("Error pinning memory");

    
    
    // auxiliary variables
    Point3D source, deltaU, deltaV, uvOrigin;
    Point3D* projParamsArrayHost;
    cudaMallocHost((void**)&projParamsArrayHost,4*PROJ_PER_BLOCK*sizeof(Point3D));
    cudaCheckErrors("Error allocating auxiliary constant memory");
    
    // Create Streams for overlapping memcopy and compute
    int nStreams=deviceCount*2;
    cudaStream_t* stream=(cudaStream_t*)malloc(nStreams*sizeof(cudaStream_t));;
    
    
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(gpuids[dev]);
        for (int i = 0; i < 2; ++i){
            cudaStreamCreate(&stream[i+dev*2]);
            
        }
    }
    cudaCheckErrors("Stream creation fail");

    int nangles_device=(nangles+deviceCount-1)/deviceCount;
    int nangles_last_device=(nangles-(deviceCount-1)*nangles_device);
    unsigned int noOfKernelCalls = (nangles_device+PROJ_PER_BLOCK-1)/PROJ_PER_BLOCK;  // We'll take care of bounds checking inside the loop if nalpha is not divisible by PROJ_PER_BLOCK
    unsigned int noOfKernelCallsLastDev = (nangles_last_device+PROJ_PER_BLOCK-1)/PROJ_PER_BLOCK; // we will use this in the memory management.
    int projection_this_block;
    cudaTextureObject_t *texImg = new cudaTextureObject_t[deviceCount];
    cudaArray **d_cuArrTex = new cudaArray*[deviceCount];
    
    for (unsigned int sp=0;sp<splits;sp++){
        
        // Create texture objects for all GPUs
        
        
        size_t linear_idx_start;
        //First one should always be  the same size as all the rest but the last
        linear_idx_start= (size_t)sp*(size_t)geoArray[0].nVoxelX*(size_t)geoArray[0].nVoxelY*(size_t)geoArray[0].nVoxelZ;
        
        
        CreateTexture(gpuids,&img[linear_idx_start],geoArray[sp],d_cuArrTex,texImg,!sp);
        cudaCheckErrors("Texture object creation fail");
        
        
        // Prepare kernel lauch variables
        
        int divU,divV;
        divU=PIXEL_SIZE_BLOCK;
        divV=PIXEL_SIZE_BLOCK;
        dim3 grid((geoArray[sp].nDetecU+divU-1)/divU,(geoArray[0].nDetecV+divV-1)/divV,1);
        dim3 block(divU,divV,PROJ_PER_BLOCK);
        
        unsigned int proj_global;
        // Now that we have prepared the image (piece of image) and parameters for kernels
        // we project for all angles.
        for (unsigned int i=0; i<noOfKernelCalls; i++) {
            for (dev=0;dev<deviceCount;dev++){
                cudaSetDevice(gpuids[dev]);
                
                for(unsigned int j=0; j<PROJ_PER_BLOCK; j++){
                    proj_global=(i*PROJ_PER_BLOCK+j)+dev*nangles_device;
                    if (proj_global>=nangles)
                        break;
                    if ((i*PROJ_PER_BLOCK+j)>=nangles_device)
                        break;
                    geoArray[sp].alpha=angles[proj_global*3];
                    geoArray[sp].theta=angles[proj_global*3+1];
                    geoArray[sp].psi  =angles[proj_global*3+2];
                    
                    
                    //precomute distances for faster execution
                    //Precompute per angle constant stuff for speed
                    computeDeltas_Siddon(geoArray[sp],proj_global, &uvOrigin, &deltaU, &deltaV, &source);
                    //Ray tracing!
                    projParamsArrayHost[4*j]=uvOrigin;		// 6*j because we have 6 Point3D values per projection
                    projParamsArrayHost[4*j+1]=deltaU;
                    projParamsArrayHost[4*j+2]=deltaV;
                    projParamsArrayHost[4*j+3]=source;
                    
                }
                cudaMemcpyToSymbolAsync(projParamsArrayDev, projParamsArrayHost, sizeof(Point3D)*4*PROJ_PER_BLOCK,0,cudaMemcpyHostToDevice,stream[dev*2]);
                cudaStreamSynchronize(stream[dev*2]);
                cudaCheckErrors("kernel fail");
                kernelPixelDetector<<<grid,block,0,stream[dev*2]>>>(geoArray[sp],dProjection[(i%2)+dev*2],i,nangles_device,texImg[dev]);
            }


            // Now that the computation is happening, we need to either prepare the memory for
            // combining of the projections (splits>1) and start removing previous results.
            
            
            // If our image does not fit in memory then we need to make sure we accumulate previous results too.
            // This is done in 2 steps: 
            // 1)copy previous results back into GPU 
            // 2)accumulate with current results
            // The code to take them out is the same as when there are no splits needed
            if( !fits_in_memory&&sp>0)
            {
                // 1) grab previous results and put them in the auxiliary variable dProjection_accum
                for (dev = 0; dev < deviceCount; dev++)
                {
                    cudaSetDevice(gpuids[dev]);
                    //Global index of FIRST projection on this set on this GPU
                    proj_global=i*PROJ_PER_BLOCK+dev*nangles_device;
                    if(proj_global>=nangles) 
                        break;

                    // Unless its the last projection set, we have PROJ_PER_BLOCK angles. Otherwise...
                    if(i+1==noOfKernelCalls) //is it the last block?
                        projection_this_block=min(nangles_device-(noOfKernelCalls-1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
                                                  nangles-proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)
                    else
                        projection_this_block=PROJ_PER_BLOCK;

                    cudaMemcpyAsync(dProjection_accum[(i%2)+dev*2], result[proj_global], projection_this_block*geo.nDetecV*geo.nDetecU*sizeof(float), cudaMemcpyHostToDevice,stream[dev*2+1]);
                }
                //  2) take the results from current compute call and add it to the code in execution.
                for (dev = 0; dev < deviceCount; dev++)
                {
                    cudaSetDevice(gpuids[dev]);
                    //Global index of FIRST projection on this set on this GPU
                    proj_global=i*PROJ_PER_BLOCK+dev*nangles_device;
                    if(proj_global>=nangles) 
                        break;

                    // Unless its the last projection set, we have PROJ_PER_BLOCK angles. Otherwise...
                    if(i+1==noOfKernelCalls) //is it the last block?
                        projection_this_block=min(nangles_device-(noOfKernelCalls-1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
                                                  nangles-proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)
                    else
                        projection_this_block=PROJ_PER_BLOCK;

                    cudaStreamSynchronize(stream[dev*2+1]); // wait until copy is finished
                    vecAddInPlace<<<(geo.nDetecU*geo.nDetecV*projection_this_block+MAXTREADS-1)/MAXTREADS,MAXTREADS,0,stream[dev*2]>>>(dProjection[(i%2)+dev*2],dProjection_accum[(i%2)+dev*2],(unsigned long)geo.nDetecU*geo.nDetecV*projection_this_block);
                }
            } // end accumulation case, where the image needs to be split 

            // Now, lets get out the projections from the previous execution of the kernels.
            if (i>0){
                for (dev = 0; dev < deviceCount; dev++)
                {
                    cudaSetDevice(gpuids[dev]);
                    //Global index of FIRST projection on previous set on this GPU
                    proj_global=(i-1)*PROJ_PER_BLOCK+dev*nangles_device;
                    if (dev+1==deviceCount) {    //is it the last device?
                        // projections assigned to this device is >=nangles_device-(deviceCount-1) and < nangles_device
                        if (i-1 < noOfKernelCallsLastDev) {
                            // The previous set(block) was not empty.
                            projection_this_block=min(PROJ_PER_BLOCK, nangles-proj_global);
                        }
                        else {
                            // The previous set was empty.
                            // This happens if deviceCount > PROJ_PER_BLOCK+1.
                            // e.g. PROJ_PER_BLOCK = 9, deviceCount = 11, nangles = 199.
                            // e.g. PROJ_PER_BLOCK = 1, deviceCount =  3, nangles =   7.
                            break;
                        }
                    }
                    else {
                        projection_this_block=PROJ_PER_BLOCK;
                    }
                    cudaMemcpyAsync(result[proj_global], dProjection[(int)(!(i%2))+dev*2],  projection_this_block*geo.nDetecV*geo.nDetecU*sizeof(float), cudaMemcpyDeviceToHost,stream[dev*2+1]);
                }
            }
            // Make sure Computation on kernels has finished before we launch the next batch.
            for (dev = 0; dev < deviceCount; dev++){
                cudaSetDevice(gpuids[dev]);
                cudaStreamSynchronize(stream[dev*2]);
            }
        }
        
        
         // We still have the last set of projections to get out of GPUs
        for (dev = 0; dev < deviceCount; dev++)
        {
            cudaSetDevice(gpuids[dev]);
            //Global index of FIRST projection on this set on this GPU
            proj_global=(noOfKernelCalls-1)*PROJ_PER_BLOCK+dev*nangles_device;
            if(proj_global>=nangles) 
                break;
            // How many projections are left here?
            projection_this_block=min(nangles_device-(noOfKernelCalls-1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
                                      nangles-proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)

            cudaDeviceSynchronize(); //Not really necessary, but just in case, we los nothing. 
            cudaCheckErrors("Error at copying the last set of projections out (or in the previous copy)");
            cudaMemcpyAsync(result[proj_global], dProjection[(int)(!(noOfKernelCalls%2))+dev*2], projection_this_block*geo.nDetecV*geo.nDetecU*sizeof(float), cudaMemcpyDeviceToHost,stream[dev*2+1]);
        }
        // Make sure everyone has done their bussiness before the next image split:
        cudaDeviceSynchronize();
    } // End image split loop.
    
    cudaCheckErrors("Main loop  fail");
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            cudaDestroyTextureObject(texImg[dev]);
            cudaFreeArray(d_cuArrTex[dev]);
    }
    delete[] texImg; texImg = 0;
    delete[] d_cuArrTex; d_cuArrTex = 0;
    // Freeing Stage
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(gpuids[dev]);
        cudaFree(dProjection[dev*2]);
        cudaFree(dProjection[dev*2+1]);
        
    }
    free(dProjection);
    
    if(!fits_in_memory){
        for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(gpuids[dev]);
            cudaFree(dProjection_accum[dev*2]);
            cudaFree(dProjection_accum[dev*2+1]);
            
        }
        free(dProjection_accum);
    }
    freeGeoArray(splits,geoArray);
    cudaFreeHost(projParamsArrayHost);
   
    
    for (int i = 0; i < nStreams; ++i)
        cudaStreamDestroy(stream[i]) ;
#ifndef NO_PINNED_MEMORY
    if (isHostRegisterSupported & (splits>1 |deviceCount>1)){
        cudaHostUnregister(img);
    }
    cudaCheckErrors("cudaFree  fail");
#endif
    //cudaDeviceReset();
    return 0;
}



int main(){
    DicomReader dicomReader;
    auto [geometry, volume] = dicomReader.readDirectory("/home/zain/Downloads/ffe79971-17a080e0-0092e608-f54c9108-89d4cb85/74640d444c104e879aa62d2949bd98b8 Anonymized479/Unknown Study/CT");
    std::cout << "Volume shape: " << volume.sizes() << std::endl;
    std::cout << "Volume min: " << volume.min().item<float>() << std::endl;
    std::cout << "Volume max: " << volume.max().item<float>() << std::endl;
    std::cout << "Volume mean: " << volume.mean().item<float>() << std::endl;
    std::cout << "Volume std: " << volume.std().item<float>() << std::endl;
    std::cout << "Volume size: " << volume.numel() << std::endl;
    std::cout << "Volume dtype: " << volume.dtype() << std::endl;
    std::cout << "Volume device: " << volume.device() << std::endl;
    std::cout << "Volume data: " << volume.data_ptr<float>() << std::endl;
    std::cout << "Volume data size: " << sizeof(float) * volume.numel() << std::endl;
    std::cout << "Volume data pointer: " << volume.data_ptr() << std::endl;
    std::cout << "Volume data pointer size: " << sizeof(float) * volume.numel() << std::endl;
    std::cout << "Volume data pointer address: " << &volume << std::endl;
    std::cout << "Volume data pointer address size: " << sizeof(float) * volume.numel() << std::endl;
}
