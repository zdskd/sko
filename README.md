网上爆料平台快手抖音网红爆料视频免费观看

int main(int argc,char **argv)
{
    // set up device
    initDevice(0);
    double iStart,iElaps;
    iStart=cpuSecond();
    int nElem=1<<24;
    printf("Vector size:%d\n",nElem);
    int nByte=sizeof(float)*nElem;
    float * a_h,*b_h,*res_h,*res_from_gpu_h;
    CHECK(cudaHostAlloc((float**)&a_h,nByte,cudaHostAllocDefault));
    CHECK(cudaHostAlloc((float**)&b_h,nByte,cudaHostAllocDefault));
    CHECK(cudaHostAlloc((float**)&res_h,nByte,cudaHostAllocDefault));
    CHECK(cudaHostAlloc((float**)&res_from_gpu_h,nByte,cudaHostAllocDefault));

    cudaMemset(res_h,0,nByte);
    cudaMemset(res_from_gpu_h,0,nByte);

    float *a_d,*b_d,*res_d;
    CHECK(cudaMalloc((float**)&a_d,nByte));
    CHECK(cudaMalloc((float**)&b_d,nByte));
    CHECK(cudaMalloc((float**)&res_d,nByte));

    initialData(a_h,nElem);
    initialData(b_h,nElem);

    sumArrays(a_h,b_h,res_h,nElem);
    dim3 block(512);
    dim3 grid((nElem-1)/block.x+1);


    //asynchronous calculation
    int iElem=nElem/N_SEGMENT;
    cudaStream_t stream[N_SEGMENT];
    for(int i=0;i<N_SEGMENT;i++)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    for(int i=0;i<N_SEGMENT;i++)
    {
        int ioffset=i*iElem;
        CHECK(cudaMemcpyAsync(&a_d[ioffset],&a_h[ioffset],nByte/N_SEGMENT,cudaMemcpyHostToDevice,stream[i]));
        CHECK(cudaMemcpyAsync(&b_d[ioffset],&b_h[ioffset],nByte/N_SEGMENT,cudaMemcpyHostToDevice,stream[i]));
        sumArraysGPU<<<grid,block,0,stream[i]>>>(&a_d[ioffset],&b_d[ioffset],&res_d[ioffset],iElem);
        CHECK(cudaMemcpyAsync(&res_from_gpu_h[ioffset],&res_d[ioffset],nByte/N_SEGMENT,cudaMemcpyDeviceToHost,stream[i]));
        CHECK(cudaStreamAddCallback(stream[i],my_callback,(void *)(stream+i),0));
