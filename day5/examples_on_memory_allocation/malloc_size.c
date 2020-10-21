#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include "benchmark.h"
#ifdef __APPLE__
#include <malloc/malloc.h>
#endif
#ifdef __linux__
#include <malloc.h>
#endif 
 
void demo( size_t N )
{

  int repeat        = 500;
  uint64_t * values = (uint64_t *) malloc(N * sizeof(uint64_t));
    
#ifdef __APPLE__
  
    size_t expected = malloc_size(values);
    printf("allocated %zu, reported allocation = %zu \n", N * sizeof(uint64_t), expected);
    BEST_TIME(malloc_size(values), expected, , repeat, 1, true);
    
#elif defined(__linux__)
    
    size_t expected = malloc_usable_size(values);
    printf("allocated %zu, reported allocation = %zu \n", N * sizeof(uint64_t), expected);
    BEST_TIME(malloc_usable_size(values), expected, , repeat, 1, true);
    
#endif
    
    free(values);
    
}


int main( void )
{
  printf("reported usabe_size for 1B allocated: %zu \n", malloc_usable_size(malloc(1)));
  printf("reported usabe_size for 4B allocated: %zu \n", malloc_usable_size(malloc(4)));
  printf("reported usabe_size for 8B allocated: %zu \n", malloc_usable_size(malloc(8)));
  printf("reported usabe_size for 16B allocated: %zu \n", malloc_usable_size(malloc(16)));
  printf("reported usabe_size for 32B allocated: %zu \n", malloc_usable_size(malloc(32)));
  
  for( size_t N = 10; N < 100000000; N *= 10 )
    demo(N);

  return 0;
}
