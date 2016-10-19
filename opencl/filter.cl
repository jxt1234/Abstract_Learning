__kernel void filter(__global float *input, __global float* kernelM, __global float* output, size_t kw, size_t kh, size_t kd, size_t stride, size_t iw, size_t ih, size_t od)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int ow = get_global_size(0);
    int oh = get_global_size(1);
    float sum = 0.0;
    int oi,i,j,k;
    int kbatchsize = kw*kh*kd;
    int inputBatchSize = iw*ih*kd;
    int outputBatchSize = ow*oh*od;
    __global float* input_base = input + inputBatchSize*z + x*stride + y*stride*iw;
    __global float* input_unit;
    __global float* kernel_base;
    __global float* output_base = output + outputBatchSize*z + x + y*ow;
    for (oi=0; oi<od; ++oi)
    {
        kernel_base = kernelM+kbatchsize*oi;
        input_unit = input_base+oi*stride*inputBatchSize;
        sum = 0.0;
        for (i=0;i<kd;++i)
        {
            for (j=0;j<kh; ++j)
            {
                for (k=0;k<kw;++k)
                {
                    sum += input_unit[k+j*iw+i*iw*ih]*kernel_base[k+j*kw+i*kw*kh];
                }
            }
        }
        *(output_base + ow*oh*oi) = sum;
    }
}
