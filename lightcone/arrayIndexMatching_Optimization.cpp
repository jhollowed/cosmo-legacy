#include <iostream>
#include <tgmath.h>
#include <time.h>
#include<random>
#include<algorithm>

// compile with 
// g++ -std=c++0x arrayIndexMatching_Optimization.cpp -o matching

int main(int argc, char *argv[]){
    
    int size = atoi(argv[1]);

    int arr1[size];

    for(int i=0;i<size;++i){
        arr1[i] = i;
    }
    auto rng = std::default_random_engine {};
    std::shuffle(&arr1[0], &arr1[size], rng);
    
    int arr2[size];
    std::copy(&arr1[0], &arr1[size], &arr2[0]);
    std::shuffle(&arr2[0], &arr2[size], rng);

    // brute force
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    int matchedIdx[size];
    for(int i = 0; i < size; ++i){
        for(int j = 0; j < size; ++j){
            if(arr1[i] == arr2[j]){
                matchedIdx[i] = j;
                break;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
   
    std::cout << "\n" << std::endl;
    if(size < 30){ 
        std::cout << "\n" << std::endl;
        std::cout << "Original: [";
        for(int i=0;i<size;++i)
            std::cout << arr1[i] << ", ";
        std::cout << "]" << std::endl;
        
        std::cout << "Matched Indices: [";
        for(int i=0;i<size;++i)
            std::cout << matchedIdx[i] << ", ";
        std::cout << "]" << std::endl;
        
        std::cout << "Matched: [";
        for(int i=0;i<size;++i)
            std::cout << arr2[matchedIdx[i]] << ", ";
        std::cout << "]" << std::endl;
    }

    std::cout << "Brute force took " << elapsed << " s" << std::endl;

    // binary search

    struct timespec start2, finish2;
    double elapsed2;
    clock_gettime(CLOCK_MONOTONIC, &start2);

    int sortedIndices[size];
    for(int i=0;i<size;++i)
        sortedIndices[i] = i;
    
    auto comparator = [&arr2](int a, int b){return arr2[a] < arr2[b]; };
    std::sort(&sortedIndices[0], &sortedIndices[size], comparator);
    
    float min = 0;
    int m = 0;
    int max = size-1;
    bool found = false;
    int matchMap[size];

    for(int i = 0; i < size; ++i){

       min = 0;
       max = size-1;
       found = false;

       while(min<=max){
           
           m = floor((min+max)/2); 
           if(arr2[sortedIndices[m]] < arr1[i])
               min = m+1;
           else if(arr2[sortedIndices[m]] > arr1[i])
               max = m-1;
           else{
               matchMap[i] = sortedIndices[m];
               found = true;
               break;
           }
        }
        if(found == false)
            matchMap[m] = -1;
    }


    clock_gettime(CLOCK_MONOTONIC, &finish2);
    elapsed2 = (finish2.tv_sec - start2.tv_sec);
    elapsed2 += (finish2.tv_nsec - start2.tv_nsec) / 1000000000.0;
    
    if(size < 30){ 
        std::cout << "Original 1: [";
        for(int i=0;i<size;++i)
            std::cout << arr1[i] << ", ";
        std::cout << "]" << std::endl;
        
        std::cout << "Original 2: [";
        for(int i=0;i<size;++i)
            std::cout << arr2[i] << ", ";
        std::cout << "]" << std::endl;
        
        std::cout << "Matched Indices: [";
        for(int i=0;i<size;++i)
            std::cout << matchMap[i] << ", ";
        std::cout << "]" << std::endl;
        
        std::cout << "Matched: [";
        for(int i=0;i<size;++i)
            std::cout << arr2[matchMap[i]] << ", ";
        std::cout << "]" << std::endl;
    }

    std::cout << "Binary search took " << elapsed2 << " s" << std::endl;
    std::cout << "\n" << std::endl;
    return 0;
}
