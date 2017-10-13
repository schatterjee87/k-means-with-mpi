#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <time.h>



void parallel_k_means(double *data, int *labels);
inline double get_value(double *, int, int, int);



#define NUM_CHANNELS 24
#define NUM_CLUSTERS 5
#define NUM_REPETITIONS 100

#define MYSQUARE(x) ((x) * (x))



int size, rank;
int height, localheight, width;









int main(int argc, char **argv) 
{
	int *labels;
	FILE *fp;
	MPI_File file;
	MPI_Offset offset;
	MPI_Status status;
	double start_time, end_time;
	double *data;
	int i;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Barrier(MPI_COMM_WORLD);
	start_time = MPI_Wtime(); /* Mark the beginning time */





	for(i = 0; i < NUM_REPETITIONS; i++)
	{
		/* Step 1: determine what we have to do */
		if (argc != 5)
		{
			printf("usage: %s <inputfile> <outputfile> <height> <width>\n", argv[0]);
			MPI_Finalize();
			exit(EXIT_SUCCESS);
		}
		/* argv[1] is name of the input file name, argv[2] is name of the output file name */
		/* argv[3] is height of image, argv[4] is width of image */
		height = atoi(argv[3]);
		width  = atoi(argv[4]);
		localheight  =  height / size;





		/* allocate memory for the data and the labels */
		data  =  (double *) malloc(localheight * width * NUM_CHANNELS * sizeof(double));
		labels  =  (int *) malloc(localheight * width * sizeof(int));
		if(!(data && labels))
		{
			puts("MEMORY ALLOCATION ERROR FOR THE LABELS ARRAY!!\n");
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}





		/* Step 2: read the image from file */
		MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
		offset  =  rank * localheight * width * NUM_CHANNELS * sizeof(double);
		MPI_File_seek(file, offset, MPI_SEEK_SET);
		MPI_File_read(file, data, localheight * width * NUM_CHANNELS, MPI_DOUBLE, &status);
		MPI_File_close(&file);





		/* Step 3: Perform k_means clustering over the data items */
		parallel_k_means(data, labels);


  


		/* Step 4:  write final output file */
		//printf("Process %d:  Writing output to the output file %s\n", rank, argv[2]);
		MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
		offset  =  rank * localheight * width * sizeof(int);
		MPI_File_seek(file, offset, MPI_SEEK_SET);
		MPI_File_write(file, labels, localheight * width, MPI_INT, &status);
		MPI_File_close(&file);
		//printf("Process %d:  Done!!\nProcess %d:  Output has been written to the output file %s\n\n", rank, rank, argv[2]);





		/* memory management */
		free(data);
		free(labels);
	}






	/* Mark the finishing time */
	MPI_Barrier(MPI_COMM_WORLD);
	end_time = MPI_Wtime();
	if(!rank)
	{
		fp  =  fopen("time_measurement.txt", "a");
		if(!fp)
		{
			fp  =  fopen("time_measurement.txt", "w");
			if(!fp)
			{
				puts("Cannot open file \"time_measurement.txt\"!!\n");
				MPI_Finalize();
				exit(EXIT_FAILURE);
			}
		}
		fprintf(fp, "Size of the input image = %lu\nNumber of Processors = %d\nAverage execution time  =  %.2lf microseconds.\n\n\n", (long unsigned int)(height * width), size, (((end_time - start_time) * 1e06) / NUM_REPETITIONS));

		fclose(fp);
	}


	MPI_Finalize();
	return 0;
}











void parallel_k_means(double *data, int *labels)
{
	const double precision  =  1e-06; /* required precision */
	register int h1, h2, i, j, k; /* loop counters */
	time_t t;
	double old_error, local_error, global_error; /* sum of squared euclidean distance */
	double *c, *local_c1, *global_c1;  /* centroids */
	int *local_counts, *global_counts;   /* size of each cluster */
	double distance, min_distance;







	/* allocate memory */
	local_counts  =  (int *) malloc(NUM_CLUSTERS * sizeof(int));
	global_counts  =  (int *) malloc(NUM_CLUSTERS * sizeof(int));
	c  =  (double *) malloc(NUM_CLUSTERS * NUM_CHANNELS * sizeof(double));
	local_c1  =  (double *) malloc(NUM_CLUSTERS * NUM_CHANNELS * sizeof(double));
	global_c1  =  (double *) malloc(NUM_CLUSTERS * NUM_CHANNELS * sizeof(double));
	if (!(local_counts && global_counts && c && local_c1 && global_c1))
	{
		printf("parallel_k_means: error allocating memory\n");
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}





	/* pick NUM_CLUSTERS points as initial centroids */
	if(!rank)
		for (k = i = 0; i < NUM_CLUSTERS; k += width/NUM_CLUSTERS, i++)
			for (j = 0; j < NUM_CHANNELS; j++)
				*(c + (i * NUM_CHANNELS) +j)   =   get_value(data, (k / width), (k % width), j);

	MPI_Bcast(c, NUM_CLUSTERS * NUM_CHANNELS, MPI_DOUBLE, 0, MPI_COMM_WORLD);






	/* main loop */
	global_error = DBL_MAX;
	do
	{
		/* save error from last step */
		old_error = global_error;
		local_error = 0;
	

		/* clear old counts and temp centroids */
		for (i = 0; i < NUM_CLUSTERS; i++)
		{
			*(local_counts + i)  =  0;
			for (j = 0; j < NUM_CHANNELS; j++)
				*(local_c1 + i*NUM_CHANNELS + j)  =  0;
		}

	
		for(h1 = 0; h1 < localheight; h1++)
			for(h2 = 0; h2 < width; h2++)
			{
				min_distance = DBL_MAX;
				/* identify the closest cluster */
				for(i = 0; i < NUM_CLUSTERS; i++)
				{
					distance = 0;
					for (j = 0; j < NUM_CHANNELS; j++ )
						distance   +=   MYSQUARE(get_value(data, h1, h2, j) - (*(c + (i * NUM_CHANNELS) + j)));
					if (distance < min_distance)
					{
						*(labels + (h1 * width) + h2)  =  i;
						min_distance = distance;
					}
				}

				/* update size and temp centroid of the destination cluster */
				for (j = 0; j < NUM_CHANNELS; j++)
					(*(local_c1 + ((*(labels + (h1 * width) + h2)) * NUM_CHANNELS) + j))   +=   get_value(data, h1, h2, j);
				(*(local_counts  +  (*(labels + (h1 * width) + h2))))++;
	    
				/* update standard error */
				local_error += min_distance;
			}

    


		/* Combine data from the different processes with the help of MPI Collective Operations */ 
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(local_c1, global_c1, NUM_CLUSTERS * NUM_CHANNELS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(local_counts, global_counts, NUM_CLUSTERS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);



		for (i = 0; i < NUM_CLUSTERS; i++) /* update all centroids */
			for (j = 0; j < NUM_CHANNELS; j++)
				*(c + (i * NUM_CHANNELS) + j)  =  global_counts[i]  ?  (*(global_c1 + (i * NUM_CHANNELS) + j) / global_counts[i]) : 0;

	
	}while(MYSQUARE(global_error - old_error) > MYSQUARE(precision));






	/* Memory Management */
	free(c);
	free(local_c1);
	free(global_c1);
	free(local_counts);
	free(global_counts);
}









inline double get_value(double *array, int i, int j, int k)
{
	return (*(array + (i * width * NUM_CHANNELS) + (j * NUM_CHANNELS) + k));
}
