Implementation of the fCMA Algorithm for large-scale noisy optimization.
This is a slightly simplified version that uses a slightly different schedule for computing the noise-learning-rate. It also does update the noise-rate in each iteration, and instead of testing, the noise-handling
can just be turned off. Otherwise it is the same as

	Oswin Krause. 2019. Large-scale noise-resilient evolution-strategies.
	In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19), Manuel López-Ibáñez (Ed.). ACM, 682-690. 
	DOI: https://doi.org/10.1145/3321707.3321724 
	
See example.py for an example of the usage, either using the explicit learning-rate schedule of the algorithm or handling it implicitely by the user(where keeping the rate as 1 amounts to a fixed learning rate
over the course of the runtime and premature convergence as usual for SGD-type algorithms).
