bool already_closed(state_t *buffer) {
  if (!buffer->isopen) {
    pthread_mutex_unlock(&buffer->chmutex);
    return true;
  }
  return false;
}

  return status;
}

void allocate2()
{
  int i;
  int *p;
  for (i=1 ; i<300000 ; i++)
  {
    p = malloc(10000 * sizeof(int));
    free (p);
  }
}

  if (already_closed(buffer)) return CLOSED_ERROR;
  size_t msg_size = my_get_msg_size(data);

  // Send signal on success
  if (BUFFER_SUCCESS == (status = buffer_add_Q(buffer, data))) {
    pthread_cond_signal(&buffer->chconrec);
  }

  if (already_closed(buffer)) return CLOSED_ERROR;

/// , zvx5107@psu.edu
/// , @psu.edu
/// , @psu.edu

void allocate(int count,int r)
{
  int x[100];
  char *b;
  r++;
  if(count >= 0)
  {
    b = malloc(pow(10,r)*sizeof(int));
    for (int i = 0 ;i<sizeof(b) ; i++)
    {
      char *ch1;
      ch1 = &b[i];
      memset(ch1,'*',sizeof(b[0])+i);
    }

enum buffer_status buffer_receive(state_t *buffer, void **data) {
  enum buffer_status status;

int main(int argc, char const *argv[])
{
	int count = 0;

  // unlock buffer and mutex for close
  pthread_mutex_unlock(&buffer->chmutex);
  pthread_mutex_unlock(&buffer->chclose);

    pid = atoi(argv[1]);        /* PID of target process */

  if (!buffer->isopen) {
    // already closed, unlock in reverse order
    pthread_mutex_unlock(&buffer->chmutex);
    pthread_mutex_unlock(&buffer->chclose);
    return CLOSED_ERROR;
  }
  buffer->isopen = false;

  return BUFFER_SUCCESS;

#include "buffer.h"

    // the thread is woken up, sanity-check the buffer
    if (already_closed(buffer)) return CLOSED_ERROR;
  }

  // lock mutex for close and buffer
  pthread_mutex_lock(&buffer->chclose);
  pthread_mutex_lock(&buffer->chmutex);

  pthread_mutex_unlock(&buffer->chmutex);

/// Because the teaching team is not capable of using Clang-Tidy
///  or realize that this is a narrowing conversion, we have our own version
///  of get_msg_size
size_t my_get_msg_size(char *data) {
  return sizeof(int) + strlen(data) + 1;
}

int x[5] = {1,2,3,4,5};

enum buffer_status buffer_destroy(state_t *buffer) {
  if (buffer->isopen) {
    return DESTROY_ERROR;
  }
  if (pthread_mutex_destroy(&buffer->chclose)
      && pthread_cond_destroy(&buffer->chconsend)
      && pthread_mutex_destroy(&buffer->chmutex)
      && pthread_cond_destroy(&buffer->chconrec)) {
    puts("Failed to destroy buffer");
  }
  fifo_free(buffer->fifoQ);
  free(buffer);
  return BUFFER_SUCCESS;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

    if (prlimit(pid, RLIMIT_STACK, NULL, &old) == -1)
        errExit("prlimit-2");
    printf("Stack limits: soft=%lld; hard=%lld\n",
            (long long) old.rlim_cur, (long long) old.rlim_max);

    printf("Stack Address = %p     Heap Address = %p  \n\n",x,b);
    allocate(count-1,r);
  }
}

		*count += 1;

  // receive messages
  if (BUFFER_SUCCESS == (status = buffer_remove_Q(buffer, data))) {
    // check whether is special message
    if (strcmp((const char *) *data, "splmsg") == 0) {
      puts("Special message. What?");
      status = BUFFER_SPECIAL_MESSSAGE;
    }

// Closes the buffer and informs all the blocking send/receive/select calls to return with CLOSED_ERROR
// Once the buffer is closed, send/receive/select operations will cease to function and just return CLOSED_ERROR
// Returns BUFFER_SUCCESS if close is successful,
// CLOSED_ERROR if the buffer is already closed, and
// BUFFER_ERROR in any other error case
enum buffer_status buffer_close(state_t *buffer) {

int main(int argc, char const *argv[]) {
  int count = 10;

// Frees all the memory allocated to the buffer , using own version of sem flags
// The caller is responsible for calling buffer_close and waiting for all threads to finish their tasks before calling buffer_destroy
// Returns BUFFER_SUCCESS if destroy is successful,
// DESTROY_ERROR if buffer_destroy is called on an open buffer, and
// BUFFER_ERROR in any other error case

void allocate1()
{
  int i;
  int *p;
  for (i=1 ; i<10000 ; i++)
  {
    p = malloc(1000 * sizeof(int));
    if(i & 1)
      free (p);
  }
}

void allocate(int count)
{
  int x[300000];
  char *c;

  // wait for messages
  while (buffer->fifoQ->avilSize >= buffer->fifoQ->size) {
    pthread_cond_wait(&buffer->chconrec, &buffer->chmutex);

void allocate()
{
    int i;
    int *p;
    for(i =1 ; i<1000000 ; i++)
    {
      p = malloc(500 * sizeof(int));
      if(func(i)) { free (p);}
    }
}

#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{
    struct rlimit old, new;
    struct rlimit *newp;
    pid_t pid;

// Writes data to the given buffer
// This is a blocking call i.e., the function only returns on a successful completion of send
// In case the buffer is full, the function waits till the buffer has space to write the new data
// Returns BUFFER_SUCCESS for successfully writing data to the buffer,
// CLOSED_ERROR if the buffer is closed, and
// BUFFER_ERROR on encountering any other generic error of any sort
enum buffer_status buffer_send(state_t *buffer, void *data) {
  enum buffer_status status;
  pthread_mutex_lock(&buffer->chmutex);

		address_print (count);
	}
}

  // wake up all threads to exit
  pthread_cond_broadcast(&buffer->chconsend);
  pthread_cond_broadcast(&buffer->chconrec);

// Creates a buffer with the given capacity
state_t *buffer_create(int capacity) {
  state_t *buffer = (state_t *) malloc(sizeof(state_t));
  // Make valgrind happy
  memset(buffer, 0, sizeof(state_t));
  buffer->fifoQ = (fifo_t *) malloc(sizeof(fifo_t));
  fifo_init(buffer->fifoQ, capacity);
  buffer->isopen = true;
  if (pthread_mutex_init(&buffer->chclose, NULL)
      && pthread_cond_init(&buffer->chconsend, NULL)
      && pthread_mutex_init(&buffer->chmutex, NULL)
      && pthread_cond_init(&buffer->chconrec, NULL)) {
    puts("Failed to initialize buffer");
    // Don't expect this to happen
  }
  return buffer;
}

  // Wait until there's sufficient space in the buffer
  while (msg_size >= fifo_avail_size(buffer->fifoQ)) {
    pthread_cond_wait(&buffer->chconsend, &buffer->chmutex);

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "check.h"

    if (already_closed(buffer)) return CLOSED_ERROR;
  }

		printf ("Address 1 = %p    Address 2 = %p\n", a, b);

#include "check.h"

int main(int argc, char const *argv[]) {
  int i;
  int *p;
  printf("Executing the code ......\n");
  allocate();

void address_print (int *count) {
	if (*count >= 10) {
		return;
	} else {
		int a[49500];
		char *b;

  printf("Entering Function \n\n");
  allocate(count);
  printf("Press Enter to continue\n");
  getchar();
  printf("Program Terminated Successfully\n");
  return 0;
}


    pthread_cond_signal(&buffer->chconsend);
  }

  pthread_mutex_lock(&buffer->chmutex);

  for (i=0 ; i<10000 ; i++)
  {
    p = malloc(1000 * sizeof(int));
    free (p);
  }
  printf("Program execution successfull\n");
  return 0;
}


  return status;
}
// test_send_correctness 1
// Reads data from the given buffer and stores it in the function’s input parameter, data (Note that it is a double pointer).
// This is a blocking call i.e., the function only returns on a successful completion of receive
// In case the buffer is empty, the function waits till the buffer has some data to read
// Return BUFFER_SPECIAL_MESSSAGE for successful retrieval of special data "splmsg"
// Returns BUFFER_SUCCESS for successful retrieval of any data other than "splmsg"
// CLOSED_ERROR if the buffer is closed, and
// BUFFER_ERROR on encountering any other generic error of any sort

int func(int i)
{
  int count = 0;
  for (int j = 2; j<= ceil(sqrt( (float) i)) ; j++)
  {
    
    if(i%j == 0)
      return 0;
  }
  return 1;
}


#define errExit(msg) do { perror(msg); exit(EXIT_FAILURE); \
                        } while (0)

	return 0;
}


    exit(EXIT_SUCCESS);
}


	printf("Enter anyxx key to exit\n");
	getchar();

		b = (char *) malloc (500 * sizeof(char));

	address_print (&count);

#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>

  if(count >= 0)
  {
    c = malloc (30000);
    printf("Stack Address  = %p    Heap Address = %p\n\n",x,c);
    allocate(count - 1);
  }
}

int main(int argc, char const *argv[]) {
  int count = 20;
  int r = 0;
  
  printf("Entering Function\n\n");
  allocate(count,r);
  printf("Program Terminated Successfully\n");
  return 0;
}


}