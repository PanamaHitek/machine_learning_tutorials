package com.panamahitek.utilities;

import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

public class GPUAvailabilityTest
{
    public static void main(String[] args)
    {
        // Initialize the JCudaDriver
        JCudaDriver.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);

        // Get the number of available GPUs
        int deviceCount[] = { 0 };
        JCudaDriver.cuDeviceGetCount(deviceCount);

        // Print the number of available GPUs
        System.out.println("Number of available GPUs: " + deviceCount[0]);
    }
}
